
package edu.cmu.cs.gabriel.network;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Timer;
import java.util.TimerTask;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import edu.cmu.cs.gabriel.token.TokenController;

public class ResultReceivingThread extends Thread {
	
	private static final String LOG_TAG = "ResultThread";
	
	private InetAddress remoteIP;
	private int remotePort;
	private Socket tcpSocket;	
	private boolean is_running = true;
	private DataOutputStream networkWriter;
	private DataInputStream networkReader;
	
	private Handler returnMsgHandler;
	private TokenController tokenController;
	private Timer timer = null;
	private Mat[] imgGuidances = new Mat[5];
	private int imgDisplayIdx = 0;
	

	public ResultReceivingThread(String GABRIEL_IP, int port, Handler returnMsgHandler, TokenController tokenController) {
		is_running = false;
		this.tokenController = tokenController;
		this.returnMsgHandler = returnMsgHandler;
		try {
			remoteIP = InetAddress.getByName(GABRIEL_IP);
		} catch (UnknownHostException e) {
			Log.e(LOG_TAG, "unknown host: " + e.getMessage());
		}
		remotePort = port;
	}

	@Override
	public void run() {
		this.is_running = true;
		Log.i(LOG_TAG, "Result receiving thread running");

		try {
			tcpSocket = new Socket();
			tcpSocket.setTcpNoDelay(true);
			tcpSocket.connect(new InetSocketAddress(remoteIP, remotePort), 5*1000);
			networkWriter = new DataOutputStream(tcpSocket.getOutputStream());
			networkReader = new DataInputStream(tcpSocket.getInputStream());
		} catch (IOException e) {
		    Log.e(LOG_TAG, Log.getStackTraceString(e));
			Log.e(LOG_TAG, "Error in initializing Data socket: " + e);
			this.notifyError(e.getMessage());
			this.is_running = false;
			return;
		}
		
		// Recv initial simulation information
		while(is_running == true){			
			try {
				String recvMsg = this.receiveMsg(networkReader);
				Log.v("zhuoc", recvMsg);
				this.notifyReceivedData(recvMsg);
			} catch (IOException e) {
				Log.e(LOG_TAG, e.toString());
				// Do not send error to handler, Streaming thread already sent it.
				this.notifyError(e.getMessage());				
				break;
			} catch (JSONException e) {
				Log.e(LOG_TAG, e.toString());
				this.notifyError(e.getMessage());
			}
		}
	}

	private String receiveMsg(DataInputStream reader) throws IOException {
		int retLength = reader.readInt();
		byte[] recvByte = new byte[retLength];
		int readSize = 0;
		while(readSize < retLength){
			int ret = reader.read(recvByte, readSize, retLength-readSize);
			if(ret <= 0){
				break;
			}
			readSize += ret;
		}
		String receivedString = new String(recvByte);
		return receivedString;
	}
	

	private void notifyReceivedData(String recvData) throws JSONException {	
		// convert the message to JSON
		JSONObject recvJSON = new JSONObject(recvData);;
		String result = null;
		int injectedToken = 0;
		String engineID = "";
		long frameID = -1;
		
		try{
			result = recvJSON.getString(NetworkProtocol.HEADER_MESSAGE_RESULT);
		} catch (JSONException e) {}
		try {
			injectedToken = recvJSON.getInt(NetworkProtocol.HEADER_MESSAGE_INJECT_TOKEN);
		} catch (JSONException e) {}
		try {
			frameID = recvJSON.getLong(NetworkProtocol.HEADER_MESSAGE_FRAME_ID);
		} catch (JSONException e) {}
		try {
            engineID = recvJSON.getString(NetworkProtocol.HEADER_MESSAGE_ENGINE_ID);
		} catch (JSONException e) {}
		
		/* refilling tokens */
		if (frameID != -1){
            Message msg = Message.obtain();
            msg.what = NetworkProtocol.NETWORK_RET_TOKEN;
            Bundle data = new Bundle();
            data.putLong(NetworkProtocol.HEADER_MESSAGE_FRAME_ID, frameID);
            data.putString(NetworkProtocol.HEADER_MESSAGE_ENGINE_ID, engineID);
            msg.setData(data);
            this.tokenController.tokenHandler.sendMessage(msg);
        }
        if (injectedToken > 0){
            this.tokenController.increaseTokens(injectedToken);
        }
		
		if (result != null){
		    Log.i(LOG_TAG, "Received result:" + result);
		    /* parsing result */
		    JSONObject resultJSON = new JSONObject(result);
            String ttsMessage = "";
            JSONArray targetLabel = null;
            JSONArray diffPiece = null;
            try {
                ttsMessage = resultJSON.getString("message");
                if (ttsMessage.equals("nothing"))
                    return;
                targetLabel = resultJSON.getJSONArray("target");
                diffPiece = resultJSON.getJSONArray("diff_piece");
            } catch (JSONException e) {
                Log.w(LOG_TAG, "Diff piece is null");
            }
            
            /* voice guidance */
			Message msg = Message.obtain();
			msg.what = NetworkProtocol.NETWORK_RET_RESULT;
			msg.obj = ttsMessage;
			this.returnMsgHandler.sendMessage(msg);
			
			/* visual guidance */
			int rowIdx = 0, colIdxStart = 0, colIdxEnd = 0, direction = 0;
			Mat imgTarget = JSONArray2Mat(targetLabel);
			Mat imgTargetNew = null;
			if (diffPiece != null) {
			    rowIdx = diffPiece.getInt(0);
			    colIdxStart = diffPiece.getInt(1);
			    colIdxEnd = diffPiece.getInt(2);
			    direction = diffPiece.getInt(3);
			
    			int height = (int) imgTarget.size().height;
    			int width = (int) imgTarget.size().width;
    			if ((rowIdx == 0 || rowIdx == height - 1) && direction > 0) {
    			    imgTargetNew = new Mat(new Size(width, height + 1), CvType.CV_8UC3);
    			    imgTargetNew.setTo(new Scalar(128, 128, 128));
    			    if (rowIdx == 0) {
    			        Mat tmp = imgTargetNew.submat(1, height + 1, 0, width);
    			        imgTarget.copyTo(tmp);
    			        rowIdx++;
    			    } else {
    			        imgTarget.copyTo(imgTargetNew.submat(0, height, 0, width));
    			    }
    			    height++;
    			    imgTarget = imgTargetNew;
    			}
			}
			
			if (diffPiece != null) {
    			imgGuidances[0] = enlargeAndShift(imgTarget, 300, 300, rowIdx, colIdxStart, colIdxEnd, direction, 1);
    			imgGuidances[1] = enlargeAndShift(imgTarget, 300, 300, rowIdx, colIdxStart, colIdxEnd, direction, 0.5);
    			imgGuidances[2] = enlargeAndShift(imgTarget, 300, 300, rowIdx, colIdxStart, colIdxEnd, direction, 0);
    			imgGuidances[3] = imgGuidances[2];
    	        imgGuidances[4] = imgGuidances[2];
			} else {
			    imgGuidances[0] = enlargeAndShift(imgTarget, 300, 300, 0, 0, 0, 0, 0);
                imgGuidances[1] = imgGuidances[0];
                imgGuidances[2] = imgGuidances[0];
                imgGuidances[3] = imgGuidances[0];
                imgGuidances[4] = imgGuidances[0];
			}
	        
	        imgDisplayIdx = 0;
	        if (timer == null) {
	            timer = new Timer();
	            timer.scheduleAtFixedRate(updateImage, 0, 500);
	        }
		}
	}
	
	TimerTask updateImage = new TimerTask(){
        @Override
        public void run() {
            Log.v("Timer", "Running timer task again");
            Message msg = Message.obtain();
            msg.what = NetworkProtocol.IMAGE_DISPLAY;
            msg.obj = imgGuidances[imgDisplayIdx];
            returnMsgHandler.sendMessage(msg);
            imgDisplayIdx = (imgDisplayIdx + 1) % 5;
        }
    };
	
	private Mat enlargeAndShift(Mat img, int heightMax, int widthMax, int rowIdx, int colIdxStart, int colIdxEnd, int direction, double ratio) {
	    int height = (int) img.size().height;
	    int width = (int) img.size().width;
	    
	    double[] shiftPixel = img.get(rowIdx, colIdxStart);
	    double scale1 = widthMax / width;
	    double scale2 = heightMax / height;
	    double scale = Math.min(scale1, scale2);
        int widthLarge = (int) (width * scale);
        int heightLarge = (int) (height * scale);
        Mat imgLarge = new Mat(new Size(widthMax, heightMax), CvType.CV_8UC3);
        imgLarge.setTo(new Scalar(128, 128, 128));
        Mat imgStuff = imgLarge.submat((heightMax - heightLarge) / 2, 
                    (heightMax - heightLarge) / 2 + heightLarge, 
                    (widthMax - widthLarge) / 2, 
                    (widthMax - widthLarge) / 2 + widthLarge);
        Imgproc.resize(img, imgStuff, imgStuff.size(), 0, 0, Imgproc.INTER_NEAREST);
	    if (direction == 1) {
	        Mat imgShiftFrom = imgStuff.submat((int) (rowIdx * scale), 
                    (int) ((rowIdx + 1) * scale), 
                    (int) (colIdxStart * scale), 
                    (int) ((colIdxEnd + 1) * scale));
            imgShiftFrom.setTo(new Scalar(128, 128, 128));
	        Mat imgShiftTo = imgStuff.submat((int) ((rowIdx - ratio) * scale), 
	                (int) ((rowIdx + 1 - ratio) * scale), 
	                (int) (colIdxStart * scale), 
	                (int) ((colIdxEnd + 1) * scale));
            imgShiftTo.setTo(new Scalar(shiftPixel[0], shiftPixel[1], shiftPixel[2]));
	    } else if (direction == 2) {
            Mat imgShiftFrom = imgStuff.submat((int) (rowIdx * scale), 
                    (int) ((rowIdx + 1) * scale), 
                    (int) (colIdxStart * scale), 
                    (int) ((colIdxEnd + 1) * scale));
            imgShiftFrom.setTo(new Scalar(128, 128, 128));
	        Mat imgShiftTo = imgStuff.submat((int) ((rowIdx + ratio) * scale), 
                    (int) ((rowIdx + 1 + ratio) * scale), 
                    (int) (colIdxStart * scale), 
                    (int) ((colIdxEnd + 1) * scale));
            imgShiftTo.setTo(new Scalar(shiftPixel[0], shiftPixel[1], shiftPixel[2]));
	    }
	    return imgLarge;
	}
	
	private Mat JSONArray2Mat(JSONArray jsonArray) {
        int height = 0;
        int width = 0;
        Mat img = null;
        try {
            height = jsonArray.length();
            width = jsonArray.getJSONArray(0).length();
            img = new Mat(new Size(width, height), CvType.CV_8UC3);
            for (int i = 0; i < height; i++) {
                JSONArray currentRow = jsonArray.getJSONArray(i);
                for (int j = 0; j < width; j++) {
                    switch (currentRow.getInt(j)) {
                        case 0:  img.put(i, j, new double[] {128, 128, 128});
                                 break;
                        case 1:  img.put(i, j, new double[] {255, 255, 255});
                                 break;
                        case 2:  img.put(i, j, new double[] {0, 255, 0});
                                 break;
                        case 3:  img.put(i, j, new double[] {0, 255, 255});
                                 break;
                        case 4:  img.put(i, j, new double[] {0, 0, 255});
                                 break;
                        case 5:  img.put(i, j, new double[] {255, 0, 0});
                                 break;
                        case 6:  img.put(i, j, new double[] {0, 0, 0});
                                 break;
                        case 7:  img.put(i, j, new double[] {255, 0, 255});
                                 break;
                        default: img.put(i, j, new double[] {128, 128, 128});
                                 break;
                    }
                }
            } 
        } catch (JSONException e) {
            Log.e(LOG_TAG, "Converting JSON array to Mat error");
        }
        return img;
	}

	private void notifyError(String errorMessage) {		
		Message msg = Message.obtain();
		msg.what = NetworkProtocol.NETWORK_RET_FAILED;
		msg.obj = errorMessage;
		this.returnMsgHandler.sendMessage(msg);
	}
	
	public void close() {
		this.is_running = false;
		timer.cancel();
		timer.purge();
		try {
			if(this.networkReader != null){
				this.networkReader.close();
				this.networkReader = null;
			}
		} catch (IOException e) {
		}
		try {
			if(this.networkWriter != null){
				this.networkWriter.close();
				this.networkWriter = null;
			}
		} catch (IOException e) {
		}
		try {
			if(this.tcpSocket != null){
				this.tcpSocket.shutdownInput();
				this.tcpSocket.shutdownOutput();			
				this.tcpSocket.close();	
				this.tcpSocket = null;
			}
		} catch (IOException e) {
		}
	}
}

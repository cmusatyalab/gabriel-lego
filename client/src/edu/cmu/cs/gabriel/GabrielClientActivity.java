package edu.cmu.cs.gabriel;

import java.util.Locale;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.preference.PreferenceManager;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.ImageView;
import edu.cmu.cs.gabriel.network.AccStreamingThread;
import edu.cmu.cs.gabriel.network.NetworkProtocol;
import edu.cmu.cs.gabriel.network.ResultReceivingThread;
import edu.cmu.cs.gabriel.network.VideoStreamingThread;
import edu.cmu.cs.gabriel.token.TokenController;

public class GabrielClientActivity extends Activity implements TextToSpeech.OnInitListener, SensorEventListener {
	
	private static final String LOG_TAG = "Main";
	private static final String CV_TAG = "CV";

	private static final int SETTINGS_ID = Menu.FIRST;
	private static final int EXIT_ID = SETTINGS_ID + 1;
	private static final int CHANGE_SETTING_CODE = 2;

	public static final int VIDEO_STREAM_PORT = 9098;
	public static final int ACC_STREAM_PORT = 9099;
	public static final int GPS_PORT = 9100;
	public static final int RESULT_RECEIVING_PORT = 9101;

	
	VideoStreamingThread videoStreamingThread;
	AccStreamingThread accStreamingThread;
	ResultReceivingThread resultThread;
	TokenController tokenController = null;

	private SharedPreferences sharedPref;
	private boolean hasStarted;
	private CameraPreview mPreview;

	private SensorManager mSensorManager = null;
	private Sensor mAccelerometer = null;
	protected TextToSpeech mTTS = null;
	
	private Mat imgGuidance = null;
	private Mat imgGuidanceLarge = null;
	private Bitmap imgGuidanceBitmap = null;
	
	private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(CV_TAG, "OpenCV loaded successfully");
                {
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		Log.d(LOG_TAG, "++onCreate");
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED+
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON+
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);		

		init();
	}

	private void init() {
		Log.d(LOG_TAG, "++init");
		mPreview = (CameraPreview) findViewById(R.id.camera_preview); //Zhuo: this initializes camera preview?
		mPreview.setPreviewCallback(previewCallback);
		
		// TextToSpeech.OnInitListener
		if (mTTS == null) {
			mTTS = new TextToSpeech(this, this);
		}
		if (mSensorManager == null) {
			mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
			mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
			mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
		}
		hasStarted = true;
		
		tokenController = new TokenController(Const.LATENCY_FILE);
		
        resultThread = new ResultReceivingThread(Const.GABRIEL_IP, RESULT_RECEIVING_PORT, returnMsgHandler, tokenController);
        resultThread.start();
        
        videoStreamingThread = new VideoStreamingThread(Const.GABRIEL_IP, VIDEO_STREAM_PORT, returnMsgHandler, tokenController);
        videoStreamingThread.start();
        
        accStreamingThread = new AccStreamingThread(Const.GABRIEL_IP, ACC_STREAM_PORT, returnMsgHandler, tokenController);
        accStreamingThread.start();
	}
		
	// Implements TextToSpeech.OnInitListener
	public void onInit(int status) {
		if (status == TextToSpeech.SUCCESS) {
			if (mTTS == null){
				mTTS = new TextToSpeech(this, this);
			}
			int result = mTTS.setLanguage(Locale.US);
			if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
				Log.e(LOG_TAG, "Language is not available.");
			}
		} else {
			// Initialization failed.
			Log.e(LOG_TAG, "Could not initialize TextToSpeech.");
		}
	}

	@Override
	protected void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
		Log.d(LOG_TAG, "++onResume");
	}

	@Override
	protected void onPause() {
		super.onPause();
		Log.d(LOG_TAG, "++onPause");
		this.terminate();
		Log.d(LOG_TAG, "--onPause");
	}

	@Override
	protected void onDestroy() {
		this.terminate();
		super.onDestroy();
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		Intent intent;

		switch (item.getItemId()) {
		case SETTINGS_ID:
			intent = new Intent().setClass(this, SettingsActivity.class);
			startActivityForResult(intent, CHANGE_SETTING_CODE);
			break;
		}

		return super.onOptionsItemSelected(item);
	}

	private PreviewCallback previewCallback = new PreviewCallback() {
		public void onPreviewFrame(byte[] frame, Camera mCamera) {
			if (hasStarted) {
				Camera.Parameters parameters = mCamera.getParameters();
				if (videoStreamingThread != null){
//					Log.d(LOG_TAG, "in");
					videoStreamingThread.push(frame, parameters);					
				}
			}
		}
	};

	private Handler returnMsgHandler = new Handler() {
		public void handleMessage(Message msg) {
			if (msg.what == NetworkProtocol.NETWORK_RET_FAILED) {
				Bundle data = msg.getData();
				String message = data.getString("message");
//				stopStreaming();
			}
			if (msg.what == NetworkProtocol.NETWORK_RET_RESULT) {
			    JSONObject obj;
			    String ttsMessage = "";
			    JSONArray target_label = null;
			    try {
                    obj = new JSONObject((String) msg.obj);
                    ttsMessage = obj.getString("message");
                    if (ttsMessage.equals("nothing"))
                        return;
                    target_label = obj.getJSONArray("target");
                    
                } catch (JSONException e) {
                    Log.e(LOG_TAG, "Return message cannot be loaded by JSON");
                }
			    
				if (mTTS != null){
					// Select a random hello.
					Log.i(LOG_TAG, "tts string: " + ttsMessage);
				    mTTS.setSpeechRate(1f);
                    mTTS.speak(ttsMessage, TextToSpeech.QUEUE_FLUSH, null);
				}
				// OpenCV stuff here...
				try {
    				int height = target_label.length();
    				int width = target_label.getJSONArray(0).length();
    				imgGuidance = new Mat(new Size(width, height), CvType.CV_8UC3);
    				for (int i = 0; i < height; i++) {
    				    JSONArray currentRow = target_label.getJSONArray(i);
    				    for (int j = 0; j < width; j++) {
        				    switch (currentRow.getInt(j)) {
        				        case 0:  imgGuidance.put(i, j, new double[] {128, 128, 128});
                                         break;
        				        case 1:  imgGuidance.put(i, j, new double[] {255, 255, 255});
                                         break;
        			            case 2:  imgGuidance.put(i, j, new double[] {0, 255, 0});
        			                     break;
        			            case 3:  imgGuidance.put(i, j, new double[] {0, 255, 255});
        			                     break;
        			            case 4:  imgGuidance.put(i, j, new double[] {0, 0, 255});
        			                     break;
        			            case 5:  imgGuidance.put(i, j, new double[] {255, 0, 0});
        			                     break;
        			            case 6:  imgGuidance.put(i, j, new double[] {0, 0, 0});
        			                     break;
        			            case 7:  imgGuidance.put(i, j, new double[] {255, 0, 255});
        			                     break;
        			            default: imgGuidance.put(i, j, new double[] {128, 128, 128});
        			                     break;
        			        }
    				    }
                    } 
				} catch (JSONException e) {
                    Log.e(LOG_TAG, "Converting JSON array to Mat error");
                }
				imgGuidanceLarge = new Mat(new Size(200, 200), CvType.CV_8UC3);
				Imgproc.resize(imgGuidance, imgGuidanceLarge, imgGuidanceLarge.size(), 0, 0, Imgproc.INTER_NEAREST);
				imgGuidanceBitmap = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888);
				Bitmap bmp = imgGuidanceBitmap;

		        try {
		            Utils.matToBitmap(imgGuidanceLarge, bmp);
		        } catch(Exception e) {
		            Log.e("org.opencv.samples.tutorial1", "Utils.matToBitmap() throws an exception: " + e.getMessage());
		            bmp.recycle();
		            bmp = null;
		        }
		        ImageView img = (ImageView) findViewById(R.id.guidance_image);
                img.setImageBitmap(bmp);
			}
		}
	};

	public void setDefaultPreferences() {
		// setDefaultValues will only be invoked if it has not been invoked
		PreferenceManager.setDefaultValues(this, R.xml.preferences, false);
		sharedPref = PreferenceManager.getDefaultSharedPreferences(this);

		sharedPref.edit().putBoolean(SettingsActivity.KEY_PROXY_ENABLED, true);
		sharedPref.edit().putString(SettingsActivity.KEY_PROTOCOL_LIST, "UDP");
		sharedPref.edit().putString(SettingsActivity.KEY_PROXY_IP, "128.2.213.25");
		sharedPref.edit().putInt(SettingsActivity.KEY_PROXY_PORT, 8080);
		sharedPref.edit().commit();
	}

	public void getPreferences() {
		sharedPref = PreferenceManager.getDefaultSharedPreferences(this);
		String sProtocol = sharedPref.getString(SettingsActivity.KEY_PROTOCOL_LIST, "UDP");
		String[] sProtocolList = getResources().getStringArray(R.array.protocol_list);
	}
	
	private void terminate() {
		Log.d(LOG_TAG, "on terminate");
		// change only soft state
		
		if ((resultThread != null) && (resultThread.isAlive())) {
			resultThread.close();
			resultThread = null;
		}
		if ((videoStreamingThread != null) && (videoStreamingThread.isAlive())) {
			videoStreamingThread.stopStreaming();
			videoStreamingThread = null;
		}
		if ((accStreamingThread != null) && (accStreamingThread.isAlive())) {
			accStreamingThread.stopStreaming();
			accStreamingThread = null;
		}
		if (tokenController != null){
			tokenController.close();
			tokenController = null;
		}
		
		// Don't forget to shutdown!
		if (mTTS != null) {
			mTTS.stop();
			mTTS.shutdown();
			mTTS = null;
			Log.d(LOG_TAG, "TTS is closed");
		}
		if (mPreview != null) {
			mPreview.setPreviewCallback(null);
			mPreview.close();
			mPreview = null;
		}
		if (mSensorManager != null) {
			mSensorManager.unregisterListener(this);
			mSensorManager = null;
			mAccelerometer = null;
		}
	}

	@Override
	public void onAccuracyChanged(Sensor sensor, int accuracy) {
	}

	@Override
	public void onSensorChanged(SensorEvent event) {
		if (event.sensor.getType() != Sensor.TYPE_ACCELEROMETER)
			return;
		if (accStreamingThread != null) {
//			accStreamingThread.push(event.values);
		}
		// Log.d(LOG_TAG, "acc_x : " + mSensorX + "\tacc_y : " + mSensorY);
	}
}

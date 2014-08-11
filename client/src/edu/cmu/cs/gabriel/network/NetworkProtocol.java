package edu.cmu.cs.gabriel.network;

public class NetworkProtocol {

	public static final int NETWORK_RET_FAILED = 1;
	public static final int NETWORK_RET_RESULT = 2;
	public static final int NETWORK_RET_CONFIG = 3;
	public static final int NETWORK_RET_TOKEN = 4;
	public static final int IMAGE_DISPLAY = 5;
	
	public static final int ACTION_ADD = 0;
	public static final int ACTION_REMOVE = 1;
	public static final int ACTION_TARGET = 2;
	
	public static final int DIRECTION_UP = 1;
    public static final int DIRECTION_DOWN = 2;
    public static final int DIRECTION_NONE = 0;
	
	public static final String HEADER_MESSAGE_CONTROL = "control";
	public static final String HEADER_MESSAGE_RESULT = "result";
	public static final String HEADER_MESSAGE_INJECT_TOKEN = "token_inject";
	public static final String HEADER_MESSAGE_FRAME_ID = "id";
	public static final String HEADER_MESSAGE_ENGINE_ID = "engine_id";

}

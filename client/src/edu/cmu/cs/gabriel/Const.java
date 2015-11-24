package edu.cmu.cs.gabriel;

import java.io.File;

import android.os.Environment;

public class Const {
	/* 
	 * Experiement variable
	 */
	
	public static final boolean IS_EXPERIMENT = false;
	
	// Transfer from the file list
	// If TEST_IMAGE_DIR is not none, transmit from the image
	public static File ROOT_DIR = new File(Environment.getExternalStorageDirectory() + File.separator + "Gabriel" + File.separator);
	public static File TEST_IMAGE_DIR = null;
	//public static File TEST_IMAGE_DIR = new File (ROOT_DIR.getAbsolutePath() + File.separator + "images-user-study-benchmark" + File.separator);	
	
	// control VM
	public static String GABRIEL_IP = "128.2.213.119";	// Cloudlet
	
	// Token
	public static int MAX_TOKEN_SIZE = 1;
	
	// image size and frame rate
	public static int MIN_FPS = 5;
	public static int IMAGE_WIDTH = 640;
	public static int IMAGE_HEIGHT = 360;

	// Result File
	public static String LATENCY_FILE_NAME = "latency-" + GABRIEL_IP + "-" + MAX_TOKEN_SIZE + ".txt";
	public static File LATENCY_DIR = new File(ROOT_DIR.getAbsolutePath() + File.separator + "exp");
	public static File LATENCY_FILE = new File (LATENCY_DIR.getAbsolutePath() + File.separator + LATENCY_FILE_NAME);
	
	public static boolean IS_USER_STUDY_BENCHMARKING = false;
	public static String USER_STUDY_BENCHMARK_FILE_NAME = LATENCY_DIR.getAbsolutePath() + File.separator + "user-study-benchmark.txt";
}

package myStack;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class ImportsData {
	public void splite(String name,int numofclass) throws Exception {
		String inputfile = "C:\\Users\\wwwcu\\Desktop\\data\\" + name + ".arff";
		File trainclass=new File("D:\\knnsearch\\trainclass.txt");
		File testclass=new File("D:\\knnsearch\\testclass.txt");
		FileWriter outrcla = new FileWriter(trainclass);
		FileWriter outecla = new FileWriter(testclass);
		
		
		FileReader fr = new FileReader(inputfile);
		BufferedReader br = new BufferedReader(fr);
		Instances data = new Instances(br);
		int trainSize = (int) Math.round(data.numInstances() * 66 / 100); //66%×÷ÎªÑµÁ·¼¯
	    int testSize = data.numInstances() - trainSize;
	    Instances train = new Instances(data, 0, trainSize);
	    Instances test = new Instances(data, trainSize, testSize);
	    ArffSaver saver1 = new ArffSaver();
		ArffSaver saver2 = new ArffSaver();
		 
		saver1.setFile(new File("C:\\Users\\wwwcu\\Desktop\\data\\"+ name + "-train.arff"));
		saver2.setFile(new File("C:\\Users\\wwwcu\\Desktop\\data\\"+ name +"-test.arff"));
		
	     
	     saver1.setInstances(train);
		 saver2.setInstances(test);
	     saver1.writeBatch();
	     saver2.writeBatch();
	     
	     int first_Class_loc = train.numAttributes()-numofclass;
	     
	     
		 int[][] trainout = filter(train,first_Class_loc,numofclass);
		 int[][] testout = filter(test,first_Class_loc,numofclass);
	     
		    for(int i=0;i<trainout.length;i++) {
		    	for(int j = 0;j<numofclass;j++) {
		    		outrcla.write(trainout[i][j]+",");
		    	}
		    	outrcla.write("\r\n");
		    }
		    outrcla.close();
		    
		    for(int i=0;i<testout.length;i++) {
		    	for(int j = 0;j<numofclass;j++) {
		    		outecla.write(testout[i][j]+",");
		    	}
		    	outecla.write("\r\n"); 
		    }
		    outecla.close();
   
	}
	public int[][] filter(Instances data,int first_Class_loc,int classnum)
	{
		int datanum = data.numInstances();

	    int[][] dataout = new int[datanum][classnum];
	    for(int i = 0;i < datanum; i++) 
	    {
	    	int NumValues = data.instance(i).numValues();
	    	for(int j = 0; j < NumValues;j++) {
		    	int tmp = data.instance(i).index(j);
		    	if(tmp<=first_Class_loc)
		    		continue;
		    	else
		    	dataout[i][tmp-first_Class_loc] = 1;
	    	}
	    }
	    return dataout;
	}
	
	public int[][] filter(Instances data,int first_Class_loc,int classnum,String s){
		int datanum = data.numInstances();
		int[][] dataout = new int[datanum][classnum];
		for(int i = 0;i<data.numInstances();i++) {
			for(int j=0;j<classnum;j++) {
				dataout[i][j]=(int) data.instance(i).value(first_Class_loc+j);
			}
		}
		return dataout;
	}
	
	 protected  Instances getTrainingSet(int iteration,Instances m_data,int m_Seed ) throws Exception {
//		    int bagSize = (int) (m_data.numInstances());
		    Instances bagData = null;
		    Random r = new Random(m_Seed + iteration);
//		    Random r = new Random();
		 	bagData = m_data.resample(r);

		    return bagData;
		  }
	
}

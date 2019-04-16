package myStack;

import weka.core.Instances;

public class DataTransform {
	String classes=new String();
	/**
	 * @Description:
	 * 将多标签数据集原有的类标签合成为一个label（字符串）
	 * 方便后来的计算最近邻操作
	 * @param data 需要删除的数据集
	 * @param index 类标签起始位置
	 * @param NumOfClass 类标签个数
	 * @return 合并类标签后的数据集
	 * @author Cui
	 */
	public void delete(Instances data , int index , int NumOfClass ) {   
		int numofAttr = data.numAttributes()-1;
		for(int i=0;i<data.numInstances();i++) {
			int trindex = index;
			for(int j=0;j<NumOfClass;j++) {
 			int temp=(int)(data.instance(i).value(trindex++));
			classes+=temp;
			}
//		System.out.println(data.instance(i).value(279));
		//Double cla=Double.parseDouble(classes);
		data.instance(i).setValue(numofAttr, classes);   //向最后一列赋新的label值
		classes="";
		//cla=0;
		}
	}

	public int[][] filter(Instances data)
	{
		int datanum = data.numInstances();

	    int NumofAtt = data.numAttributes();

	    int[][] dataout = new int[datanum][NumofAtt];
	    for(int i = 0;i < datanum; i++) 
	    {
	    	int NumValues = data.instance(i).numValues();
	    	for(int j = 0; j < NumValues;j++) {
		    	int tmp = data.instance(i).index(j);
		    	dataout[i][tmp] = 1;
	    	}
	    }
	    return dataout;
	}
	
	public boolean[] toBool(double[] data) {
		boolean[] bipartition = new boolean[data.length];
		for(int i=0;i<data.length;i++) {
			if(data[i] == 1.0)
				bipartition[i] = true;
			else
				bipartition[i] = false;
		}
		return bipartition;
	}
	public double[] todouble(boolean[] data) {
		double[] bipartition = new double[data.length];
		for(int i=0;i<data.length;i++) {
			if(data[i]) {
				bipartition[i] = 1;
			} else
				bipartition[i] = 0;
		}
		return bipartition;
	}
	

}

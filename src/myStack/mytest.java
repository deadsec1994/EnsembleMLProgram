package myStack;

import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;

public class mytest {
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
	          
	        String arffFile_train = "/Users/cuiwei/experiment/data/emotions.arff";
	        String xmlFile_train ="/Users/cuiwei/experiment/data/emotions.xml";
//	        String arffFile_test = "C:\\Users\\wwwcu\\Desktop\\data\\medical-test.arff";  
//	        String xmlFile_test ="C:\\Users\\wwwcu\\Desktop\\data\\medical.xml";  
	        MultiLabelInstances data_train = null;
//	        MultiLabelInstances data_test = null;
	        data_train = new MultiLabelInstances(arffFile_train, xmlFile_train);  
//	        data_test = new MultiLabelInstances(arffFile_test, xmlFile_test);  
	        output eval = new output();
	        MultipleEvaluation results;
	        MLkNN mlknn=new MLkNN();   
	        mlknn.build(data_train);  
//	        results=eval.evaluate(mlknn,data_test,data_train);  
	        results = eval.crossValidate(mlknn, data_train,10);
	        System.out.println(results);
	        //results = eval.crossValidate(mlknn, data_train, numFolds);
	       //System.out.println(results);
	    }

}

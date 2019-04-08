package myStack;

import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;

import java.util.Random;


public class StackMeasure {

    public static void main(String[] args) throws Exception {
        // TODO Auto-generated method stub
        String arffFile_data = "/Users/cuiwei/experiment/data/yeast.arff";
        String xmlFile_data = "/Users/cuiwei/experiment/data/yeast.xml";

        MultiLabelInstances dataset = null;
        dataset = new MultiLabelInstances(arffFile_data, xmlFile_data);
        Instances workingset = dataset.getDataSet();
        workingset.randomize(new Random(1));
        Prediction p = new Prediction();
        Caculator get = new Caculator();
        int numofcla = dataset.getNumLabels();
        int[] labelIndices = dataset.getLabelIndices();


        for (int fold = 0; fold < 10; fold++) {
            System.out.println("fold:" + fold);
            Instances train = workingset.trainCV(10, fold);
            Instances test = workingset.testCV(10, fold);
            double[][] OutTestData = get.getlabels(labelIndices, test);
            double[][] OutTrainData = get.getlabels(labelIndices, train);
            int neighbours = 5;
            for (int ptime = 0; ptime < 10; ptime++) {
                Instances newdata = get.getTrainingSet(ptime, train, 1);  //抽样
                MultiLabelInstances mlTrain = new MultiLabelInstances(newdata, dataset.getLabelsMetaData());
                MLkNN mlknn = new MLkNN(neighbours,1);
                mlknn.build(mlTrain);

                OutTrainData = get.Predictionresult(mlknn, numofcla, train, OutTrainData, ptime);
                OutTestData = get.Predictionresult(mlknn, numofcla, test, OutTestData, ptime);
                neighbours +=3;
            }
            //创建新数据集保存结果

            Instances worksetTrain = get.creatnewInstance(OutTrainData);
            Instances worksetTest = get.creatnewInstance(OutTestData);


            //类标签平衡
            MyClassBalancer classfilter = new MyClassBalancer();
            Instances balan = classfilter.process(worksetTrain);




            p.Predict(balan, worksetTest, numofcla,fold);
        }

        double[] Adamesaure = p.getvalue("-A");
        double[] Bagmesaure = p.getvalue("-B");
        System.out.println("AdaBoost Accuracy:" + Adamesaure[0] + " Precision:" + Adamesaure[1] + " \nRecall:" + Adamesaure[2] +
                " HL:" + Adamesaure[3] + " F-measure：" + Adamesaure[4]);
        System.out.println("\n\nBagging Accuracy:" + Bagmesaure[0] + " Precision:" + Bagmesaure[1] + " \nRecall:" + Bagmesaure[2] +
                " HL:" + Bagmesaure[3] + " F-measure:" + Bagmesaure[4]);
    }

}

package myStack;

import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;


public class StackMeasure {

    public static void main(String[] args) throws Exception {
        // TODO Auto-generated method stub
        String arffFile_data = "/Users/cuiwei/experiment/data/emotions.arff";
        String xmlFile_data = "/Users/cuiwei/experiment/data/emotions.xml";

        MultiLabelInstances dataset = null;
        ImportsData id = new ImportsData();
        dataset = new MultiLabelInstances(arffFile_data, xmlFile_data);
        Instances workingSet = new Instances(dataset.getDataSet());
        Prediction p = new Prediction();
        Caculator get = new Caculator();
        int numofcla = dataset.getNumLabels();
        int[] labelIndices = dataset.getLabelIndices();
        int neighbour = 5;
        workingSet.randomize(new Random(1));


        for (int fold = 0; fold < 10; fold++) {
            Instances train = workingSet.trainCV(10, fold);
            Instances test = workingSet.testCV(10, fold);
            double[][] OutTestData = get.getlabels(labelIndices, test);
            double[][] OutTrainData = get.getlabels(labelIndices, train);

            for (int ptime = 0; ptime < 10; ptime++) {
                Instances newdata = get.getTrainingSet(ptime, train, 1);  //抽样
                MultiLabelInstances mlTrain = new MultiLabelInstances(newdata, dataset.getLabelsMetaData());
                MLkNN mlknn = new MLkNN(neighbour,1);
                mlknn.build(mlTrain);

                OutTrainData = get.Predictionresult(mlknn, numofcla, train, OutTrainData, ptime);
                OutTestData = get.Predictionresult(mlknn, numofcla, test, OutTestData, ptime);
                neighbour +=3;
            }
            //创建新数据集保存结果

            Instances worksetTrain = get.creatnewInstance(OutTrainData);
            Instances worksetTest = get.creatnewInstance(OutTestData);


            //数据集输出
//			 String trainpath = "/Users/cuiwei/experiment/knnsearch/scene/c="+fold+"/train.arff";
//			 String testpath = "/Users/cuiwei/experiment/knnsearch/scene/c="+fold+"/test.arff";
//			 generateArffFile(worksetTrain,trainpath);
//			 generateArffFile(worksetTest,testpath);

//            String trainpath = "/Users/cuiwei/experiment/knnsearch/scene/c=" + fold;
//            File file1dir = new File(trainpath);
//            File file1 = new File(file1dir, "AdaCount_1.txt");
//            if (!file1dir.isDirectory())
//                file1dir.mkdir();
//            if (!file1.isFile())
//                file1.createNewFile();
//            FileWriter out1 = new FileWriter(file1, true);


            //类标签平衡
            MyClassBalancer classfilter = new MyClassBalancer();
            Instances balan = classfilter.process(worksetTrain);

            p.Predict(balan, worksetTest, numofcla);

            System.out.println("fold:" + fold);
        }

        ArrayList<boolean[]> pre = p.getArray("p");
        ArrayList<boolean[]> real = p.getArray("");
        output ot = new output();
        ot.outpre(pre,real);

        double[] Adamesaure = p.getvalue("-A");
        double[] Bagmesaure = p.getvalue("-B");
        System.out.println("AdaBoost Accuracy:" + Adamesaure[0] + " Precision:" + Adamesaure[1] + " \nRecall:" + Adamesaure[2] +
                " HL:" + Adamesaure[3] + " F-measure：" + Adamesaure[4]);
        System.out.println("\n\nBagging Accuracy:" + Bagmesaure[0] + " Precision:" + Bagmesaure[1] + " \nRecall:" + Bagmesaure[2] +
                " HL:" + Bagmesaure[3] + " F-measure:" + Bagmesaure[4]);
    }

}

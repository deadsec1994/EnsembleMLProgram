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

        workingSet.randomize(new Random(1));


        for (int fold = 0; fold < 10; fold++) {
            Instances train = workingSet.trainCV(10, fold);
            Instances test = workingSet.testCV(10, fold);
            double[][] OutTestData = get.getlabels(labelIndices, test);
            double[][] OutTrainData = get.getlabels(labelIndices, train);
            int neighbour = 1;

            for (int ptime = 0; ptime < 10; ptime++) {
                Instances newdata = get.getTrainingSet(ptime, train, 1);  //抽样
                MultiLabelInstances mlTrain = new MultiLabelInstances(newdata, dataset.getLabelsMetaData());
//                MLkNN mlknn = new MLkNN(neighbour,1);
                MLkNN mlknn = new MLkNN();

                mlknn.build(mlTrain);

                OutTrainData = get.Predictionresult(mlknn, numofcla, train, OutTrainData, ptime);
                OutTestData = get.Predictionresult(mlknn, numofcla, test, OutTestData, ptime);
//                neighbour +=1;

            }
            //创建新数据集保存结果
            MLkNN mlknn = new MLkNN();
            mlknn.build(new MultiLabelInstances(train, dataset.getLabelsMetaData()));


            double[] mlknnpre = get.Predictionresult(mlknn,numofcla,test);
            Instances worksetTrain = get.creatnewInstance(OutTrainData);
            Instances worksetTest = get.creatnewInstance(OutTestData);

            //类标签平衡
            MyClassBalancer classfilter = new MyClassBalancer();
            Instances balan = classfilter.process(worksetTrain);

            p.Predict(worksetTrain, worksetTest, numofcla,fold,mlknnpre);
            //0.045341,0.025106,0.024379,0.015278,0.376713,0.174438,0.10968,0.015391,0.005591,0.270415,0
            //0.944109,0.75497,0.93934,0.940507,0.828324,0.795575,0.908092,0.752403,0.634688,0.536323,1
            System.out.println("fold:" + fold);
        }

//        ArrayList<boolean[]> pre = p.getArray("p");
//        ArrayList<boolean[]> real = p.getArray("");
//        output ot = new output();
//        ot.outpre(pre,real);

        double[] Adamesaure = p.getvalue("-A");
        double[] Bagmesaure = p.getvalue("-B");
        System.out.println("AdaBoost Accuracy:" + Adamesaure[0] + " Precision:" + Adamesaure[1] + " \nRecall:" + Adamesaure[2] +
                " HL:" + Adamesaure[3] + " F-measure：" + Adamesaure[4]);
        System.out.println("\n\nBagging Accuracy:" + Bagmesaure[0] + " Precision:" + Bagmesaure[1] + " \nRecall:" + Bagmesaure[2] +
                " HL:" + Bagmesaure[3] + " F-measure:" + Bagmesaure[4]);
    }

}

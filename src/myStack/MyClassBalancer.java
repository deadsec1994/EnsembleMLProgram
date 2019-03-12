package myStack;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class MyClassBalancer {
    public Instances process(Instances instances) throws Exception {
        Instances instacescopy = instances;

        if (instacescopy.classAttribute().isNumeric()) {
            throw new Error("Class can not be Numeric!");
        }

        double[] sumOfWeightsPerClass = new double[instacescopy.numClasses()];
        Instances instanceofclass1 = new Instances(instances, 0);
        Instances instanceofclass0 = new Instances(instances, 0);

        for (int i = 0; i < instacescopy.numInstances(); i++) {
            Instance inst = instacescopy.instance(i);
            sumOfWeightsPerClass[(int) inst.classValue()] += inst.weight(); //默认权重为1，所以相当于统计每个类的数量
            if ((int) inst.classValue() == 1) {
                instanceofclass1.add(inst);
            }
            else{
                instanceofclass0.add(inst);
            }

        }
        double min, max;
        min = max = sumOfWeightsPerClass[0];
        //获得最小,最大类的数量
        for (double i : sumOfWeightsPerClass) {
            if (i < min)
                min = i;
            if (i > max)
                max = i;
        }
        Instances result = new Instances(instances, 0);

        int count = 0;

        for (int i = 0;i<instanceofclass0.numInstances();i++) {
            Instance tmp = instanceofclass0.instance(i);
            if(count<min){
                result.add(tmp);
                count++;
            }
            if (count == min) {
                count = 0;
                break;
            }
        }

        for (int j = 0;j<instanceofclass1.numInstances();j++) {
            Instance tmp = instanceofclass1.instance(j);
            if(count<min){
                result.add(tmp);
                count++;
            }
        }
//        int c1 = 0, c0 = 0;
//        for (int i = 0;i<result.numInstances();i++) {
//            Instance tmp = result.instance(i);
//            if (tmp.classValue() == 1)
//                c1++;
//            else
//                c0++;
//        }
//        System.out.println("c1:" + c1 + " ," + "c0:" + c0 );
        return result;
    }
}

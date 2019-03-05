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
        for (int i = 0; i < instacescopy.numInstances(); i++) {
            Instance inst = instacescopy.instance(i);
            sumOfWeightsPerClass[(int) inst.classValue()] += inst.weight(); //默认权重为1，所以相当于统计每个类的数量
            if ((int) inst.classValue() == 1) {
                instanceofclass1.add(inst);
                instacescopy.delete(i);
            }
        }
        double min,max;
        min = max = sumOfWeightsPerClass[0];
        //获得最小,最大类的数量
        for (double i : sumOfWeightsPerClass) {
            if (i < min)
                min = i;
            if(i>max)
                max = i;
        }
        Resample filter = new Resample();
        filter.setInputFormat(instances);
        double percent = Math.floor(min*100/max);
        String[] options = {"-Z",String.valueOf(percent)};
        filter.setOptions(options);
        Instances instanceofclass0 = Filter.useFilter(instacescopy, filter);
        Instances result = new Instances(instances, 0);
        for(Instance tmp:instanceofclass0){
            result.add(tmp);
        }
        for(Instance tmp:instanceofclass1){
            result.add(tmp);
        }
        return result;
    }
}

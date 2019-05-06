package myStack;

import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;

public class MyClassBalancer {
    /**
     * @Description TODO 少数类过采样
     * @param workset  不均衡数据集
     * @Return weka.core.Instances 平衡后的数据集
     * @Author cuiwei
     * @Date 2019-03-20 20:35
     */
    public Instances process(Instances workset,int fold) throws Exception {

        if (workset.classAttribute().isNumeric()) {
            throw new Error("Class can not be Numeric!");
        }

        double[] sumOfWeightsPerClass = new double[workset.numClasses()];
        Instances instanceofclass1 = new Instances(workset, 0);
        Instances instanceofclass0 = new Instances(workset, 0);
        for (int i = 0; i < workset.numInstances(); i++) {
            Instance inst = workset.instance(i);
            sumOfWeightsPerClass[(int) inst.classValue()] += inst.weight(); //默认权重为1，所以相当于统计每个类的数量
            if ((int) inst.classValue() == 1) {
                instanceofclass1.add(inst);
            }
            else {
                instanceofclass0.add(inst);
            }
        }


        int numofclass1 = instanceofclass1.numInstances();
        int numofclass0 = workset.numInstances()-numofclass1;
        double chouyang = Math.abs(numofclass1 - numofclass0);
        Random r = new Random(1+fold);
        for(int i = 0;i<chouyang;i++){
            if(numofclass0<numofclass1){
                workset.add(instanceofclass0.instance(r.nextInt(instanceofclass0.numInstances())));
            }else {
                workset.add(instanceofclass1.instance(r.nextInt(instanceofclass1.numInstances())));
            }
        }

        return workset;
    }
}

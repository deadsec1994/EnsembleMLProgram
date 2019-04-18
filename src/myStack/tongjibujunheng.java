package myStack;

import mulan.data.MultiLabelInstances;
import weka.core.Instances;

public class tongjibujunheng {
    public static void main(String[] args) throws Exception {
        String[] filename = {"emotions", "yeast", "scene", "medical", "Arts1", "Health1", "Computers1", "Business1"};
        Double[] percent = new Double[filename.length];
        int loc = 0;
        for (String name : filename) {
            double count = 0;
            String arffFile_data = "/Users/cuiwei/experiment/data/" + name + ".arff";
            String xmlFile_data = "/Users/cuiwei/experiment/data/" + name + ".xml";
            MultiLabelInstances dataset = null;
            dataset = new MultiLabelInstances(arffFile_data, xmlFile_data);
            int numcla = dataset.getNumLabels();
            Instances workingSet = new Instances(dataset.getDataSet());
            int[] labelIndices = dataset.getLabelIndices();
            for (int i = 0; i < workingSet.numInstances(); i++) {
                for (int j = 0; j < labelIndices.length; j++) {
                    if (workingSet.instance(i).value(labelIndices[j]) == 1)
                        ++count;
                }
            }
            double per = count / (workingSet.numInstances() * numcla);
            percent[loc++] = per;
            System.out.print(dataset.getNumInstances()+", ");
        }
        for(double ele :percent)
            System.out.println(ele+",");
    }
}

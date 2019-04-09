package myStack;

import clus.Clus;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.clus.ClusWrapperClassification;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import mulan.evaluation.measure.Measure;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.pmml.Array;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public class output extends Evaluator {
    public void outpre(ArrayList<boolean[]> pre,ArrayList<boolean[]> real) throws Exception{
        BufferedWriter bw = new BufferedWriter(new FileWriter("/Users/cuiwei/experiment/StackPre.txt"));
        bw.append("real Label ----------------------------Pre Label");
        bw.newLine();
        for(int i=0;i<real.size();i++) {
            boolean[] prea = pre.get(i);
            boolean[] rea = real.get(i);
            int[] line1 = new int[rea.length];
            int[] line2 = new int[prea.length];
            for(int j=0;j<line1.length;j++){
                line1[j] = rea[j]?1:0;
                line2[j] = prea[j]?1:0;
            }
            for(int num:line1){
                bw.append(num+",");
            }
            bw.append("                         ");
            for(int num:line2){
                bw.append(num+",");
            }
            bw.newLine();
            bw.flush();
        }
    }
        /** seed for reproduction of cross-validation results **/
        private int seed = 1;

        /**
         * Sets the seed for reproduction of cross-validation results
         *
         * @param aSeed seed for reproduction of cross-validation results
         */
        public void setSeed(int aSeed) {
            seed = aSeed;
        }

        /**
         * Evaluates a {@link MultiLabelLearner} on given test data set using specified evaluation
         * measures
         *
         * @param learner the learner to be evaluated
         * @param mlTestData the data set for evaluation
         * @param measures the evaluation measures to compute
         * @return an Evaluation object
         * @throws IllegalArgumentException if an input parameter is null
         * @throws Exception when evaluation fails
         */

        private ArrayList<boolean[]> pre = new ArrayList<>();
        private ArrayList<boolean[]> real = new ArrayList<>();

    public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances mlTestData,
                               List<Measure> measures) throws Exception {
        checkLearner(learner);
        checkData(mlTestData);
        checkMeasures(measures);

        // reset measures
        for (Measure m : measures) {
            m.reset();
        }

        int numLabels = mlTestData.getNumLabels();
        int[] labelIndices = mlTestData.getLabelIndices();
        GroundTruth truth;
        Set<Measure> failed = new HashSet<Measure>();
        Instances testData = mlTestData.getDataSet();
        int numInstances = testData.numInstances();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testData.instance(instanceIndex);
            boolean hasMissingLabels = mlTestData.hasMissingLabels(instance);
            Instance labelsMissing = (Instance) instance.copy();
            labelsMissing.setDataset(instance.dataset());
            for (int i = 0; i < mlTestData.getNumLabels(); i++) {
                labelsMissing.setMissing(labelIndices[i]);
            }
            MultiLabelOutput output = learner.makePrediction(labelsMissing);

            pre.add(output.getBipartition());//获取预测标签

            if (output.hasPvalues()) {// check if we have regression outputs
                truth = new GroundTruth(getTrueScores(instance, numLabels, labelIndices));
            } else {
                truth = new GroundTruth(getTrueLabels(instance, numLabels, labelIndices));
            }

            real.add(truth.getTrueLabels()); //获取真实标签

            Iterator<Measure> it = measures.iterator();
            while (it.hasNext()) {
                Measure m = it.next();
                if (!failed.contains(m)) {
                    try {
                        if (hasMissingLabels && !m.handlesMissingValues()) {
                            continue;
                        }
                        m.update(output, truth);
                    } catch (Exception ex) {
                        failed.add(m);
                    }
                }
            }
        }
        outpre(pre,real);//输出结果
        return new Evaluation(measures, mlTestData);
    }

    private void checkLearner(MultiLabelLearner learner) {
        if (learner == null) {
            throw new IllegalArgumentException("Learner to be evaluated is null.");
        }
    }

    private void checkData(MultiLabelInstances data) {
        if (data == null) {
            throw new IllegalArgumentException("Evaluation data object is null.");
        }
    }

    private void checkMeasures(List<Measure> measures) {
        if (measures == null) {
            throw new IllegalArgumentException("List of evaluation measures to compute is null.");
        }
    }

    private void checkFolds(int someFolds) {
        if (someFolds < 2) {
            throw new IllegalArgumentException("Number of folds must be at least two or higher.");
        }
    }


    private List<Measure> prepareMeasures(MultiLabelLearner learner,
                                          MultiLabelInstances mlTestData, MultiLabelInstances mlTrainData) {
        List<Measure> measures = new ArrayList<Measure>();

        MultiLabelOutput prediction;
        try {
            // MultiLabelLearner copyOfLearner = learner.makeCopy();
            // prediction = copyOfLearner.makePrediction(data.getDataSet().instance(0));
            prediction = learner.makePrediction(mlTestData.getDataSet().instance(0));
            int numOfLabels = mlTestData.getNumLabels();
            // add bipartition-based measures if applicable
            if (prediction.hasBipartition()) {
                // add example-based measures
                measures.add(new HammingLoss());
                measures.add(new SubsetAccuracy());
                measures.add(new ExampleBasedPrecision());
                measures.add(new ExampleBasedRecall());
                measures.add(new ExampleBasedFMeasure());
                measures.add(new ExampleBasedAccuracy());
                measures.add(new ExampleBasedSpecificity());
                // add label-based measures
                measures.add(new MicroPrecision(numOfLabels));
                measures.add(new MicroRecall(numOfLabels));
                measures.add(new MicroFMeasure(numOfLabels));
                measures.add(new MicroSpecificity(numOfLabels));
                measures.add(new MacroPrecision(numOfLabels));
                measures.add(new MacroRecall(numOfLabels));
                measures.add(new MacroFMeasure(numOfLabels));
                measures.add(new MacroSpecificity(numOfLabels));
            }
            // add ranking-based measures if applicable
            if (prediction.hasRanking()) {
                // add ranking based measures
                measures.add(new AveragePrecision());
                measures.add(new Coverage());
                measures.add(new OneError());
                measures.add(new IsError());
                measures.add(new ErrorSetSize());
                measures.add(new RankingLoss());
            }
            // add confidence measures if applicable
            if (prediction.hasConfidences()) {
                measures.add(new MeanAveragePrecision(numOfLabels));
                measures.add(new GeometricMeanAveragePrecision(numOfLabels));
                measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
                measures.add(new GeometricMeanAverageInterpolatedPrecision(numOfLabels, 10));
                measures.add(new MicroAUC(numOfLabels));
                measures.add(new MacroAUC(numOfLabels));
                measures.add(new LogLoss());
            }
            // add hierarchical measures if applicable
            if (mlTestData.getLabelsMetaData().isHierarchy()) {
                measures.add(new HierarchicalLoss(mlTestData));
            }
            // add regression measures if applicable
            if (prediction.hasPvalues()) {
                measures.add(new AverageRMSE(numOfLabels));
                measures.add(new AverageRelativeRMSE(numOfLabels, mlTrainData, mlTestData));
                measures.add(new AverageMAE(numOfLabels));
                measures.add(new AverageRelativeMAE(numOfLabels, mlTrainData, mlTestData));
            }
        } catch (Exception ex) {
            Logger.getLogger(mulan.evaluation.Evaluator.class.getName()).log(Level.SEVERE, null, ex);
        }

        return measures;
    }

    private boolean[] getTrueLabels(Instance instance, int numLabels, int[] labelIndices) {

        boolean[] trueLabels = new boolean[numLabels];
        for (int counter = 0; counter < numLabels; counter++) {
            int classIdx = labelIndices[counter];
            String classValue = instance.attribute(classIdx).value((int) instance.value(classIdx));
            trueLabels[counter] = classValue.equals("1");
        }

        return trueLabels;
    }

    private double[] getTrueScores(Instance instance, int numLabels, int[] labelIndices) {

        double[] trueScores = new double[numLabels];
        for (int counter = 0; counter < numLabels; counter++) {
            int classIdx = labelIndices[counter];
            double score;
            if (instance.isMissing(classIdx)) {// if target is missing
                score = Double.NaN; // make it equal to Double.Nan
            } else {
                score = instance.value(classIdx);
            }
            trueScores[counter] = score;
        }

        return trueScores;
    }

    private MultipleEvaluation innerCrossValidate(MultiLabelLearner learner,
                                                  MultiLabelInstances data, boolean hasMeasures, List<Measure> measures, int someFolds) {
        Evaluation[] evaluation = new Evaluation[someFolds];

        Instances workingSet = new Instances(data.getDataSet());
        workingSet.randomize(new Random(seed));
        for (int i = 0; i < someFolds; i++) {
            System.out.println("Fold " + (i + 1) + "/" + someFolds);
            try {
                Instances train = workingSet.trainCV(someFolds, i);
                Instances test = workingSet.testCV(someFolds, i);
                MultiLabelInstances mlTrain = new MultiLabelInstances(train,
                        data.getLabelsMetaData());
                MultiLabelInstances mlTest = new MultiLabelInstances(test, data.getLabelsMetaData());
                MultiLabelLearner clone = learner.makeCopy();
                clone.build(mlTrain);
                if (hasMeasures) {
                    evaluation[i] = evaluate(clone, mlTest, measures);
                } else {
                    evaluation[i] = evaluate(clone, mlTest, mlTrain);
                }
            } catch (Exception ex) {
                Logger.getLogger(mulan.evaluation.Evaluator.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        MultipleEvaluation me = new MultipleEvaluation(evaluation, data);
        me.calculateStatistics();
        return me;
    }
}


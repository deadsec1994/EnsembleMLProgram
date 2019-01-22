package myStack;


public class Measure {
	protected double sum1;
	protected double sum2;
	protected double sum3;
	protected double sum4;
    /**
     * The number of validation examples processed
     */
    protected int count1;
    protected int count2;
    protected int count3;
    protected int count4;
    
    
    protected double temp;

	public void reset() {
	        sum1 = 0;
	        count1 = 0;
	        sum2 = 0;
	        count2 = 0;
	        sum3 = 0;
	        count3 = 0;
	        sum4 = 0;
	        count4 = 0;
	    }


	
	public double getValue(String s) {
	    if(s.equals("-A"))
	    	  temp = sum1 / count1;
	    else if(s.equals("-P"))
	    	  temp = sum2 / count2;
	    else if(s.equals("-R"))
	    	  temp = sum3 / count3; 
	    else if(s.equals("-H"))
	    	temp = sum4/count4;
		return temp;
	    }
	
	
	public void Accuracy(boolean[] bipartition, boolean[] truth) {
        double intersection = 0;
        double union = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i] && truth[i]) {
                intersection++;
            }
            if (bipartition[i] || truth[i]) {
                union++;
            }
        }

        if (union == 0) {
            sum1 += 1;
        } else {
            sum1 += intersection / union;
        }
        count1++;
    }
	public void Precision(boolean[] bipartition, boolean[] truth) {
        double tp = 0;
        double fp = 0;
        double fn = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i]) {
                if (truth[i]) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                if (truth[i]) {
                    fn++;
                }
            }
        }
        if (tp + fp + fn == 0) {
        	sum2 +=  1;
        }
        if (tp + fp == 0) {
        	sum2 +=  0;
        }
        sum2 +=  precision(tp,  fp,  fn);
        count2++;
    }
    
    public void Recall(boolean[] bipartition, boolean[] truth) {
        double tp = 0;
        double fp = 0;
        double fn = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i]) {
                if (truth[i]) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                if (truth[i]) {
                    fn++;
                }
            }
        }
        if (tp + fp + fn == 0) {
        	sum3 += 1;
        }
        if (tp + fn == 0) {
        	sum3 += 0;
        }
        sum3 += recall(tp,  fp,  fn);
        count3++;
    }
    public static double precision(double tp, double fp, double fn) {
        if (tp + fp + fn == 0) {
            return 1;
        }
        if (tp + fp == 0) {
            return 0;
        }
        return tp / (tp + fp);
    }
    public static double recall(double tp, double fp, double fn) {
        if (tp + fp + fn == 0) {
            return 1;
        }
        if (tp + fn == 0) {
            return 0;
        }
        return tp / (tp + fn);
    }
    
    public void HammingLoss(boolean[] bipartition, boolean[] groundTruth) {
    	
    	 double symmetricDifference = 0;
         for (int i = 0; i < groundTruth.length; i++) {
             if (bipartition[i] != groundTruth[i]) {
                 symmetricDifference++;
             }
         }
         sum4 += symmetricDifference / groundTruth.length;
         count4++;
    }
}

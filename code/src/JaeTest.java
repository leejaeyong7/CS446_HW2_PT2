/*
 Jae Yong Lee
 University of Illinois at Urbana Champaign
 UIN: lee896
 CS 446 HW1 part 2
 */


/*
 includes
 */
import java.io.File;
import java.io.FileReader;
import java.util.Scanner;
import java.util.Collections;
import java.util.Enumeration;
import java.util.ArrayList;


import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;

/*
 class implementation
 */

public class JaeTest {

	//this will merge all data set except one given by exception index
	public static Instances mergeExcept(Instances [] ds ,int exceptionIndex)
	{
		//copies attributes from first fold (since all attributes are shared in this case)
		Instances retInst = new Instances(ds[0]);
		retInst.delete();
		for(int i = 0 ; i < 5 ; i++)
		{
			if(i != exceptionIndex)
			{
				for (Enumeration<Instance> e = ds[i].enumerateInstances(); e.hasMoreElements();)
				{
					retInst.add(e.nextElement());
				}
			}
		}
		return retInst;
	}
	
    public static void main(String[] args) throws Exception {

		// Check for valid argument (the address of a .arff file, perhaps produced by FeatureGenerator.java)
		if (args.length != 1) {
			System.err.println("Usage: JaeTest arff-file-path");
			System.exit(-1);
		}

		// Declare variables and load data.
		Instances[] datas = new Instances[]{ new Instances(new FileReader(new File(args[0]+"badges.fold1.arff"))),
											 new Instances(new FileReader(new File(args[0]+"badges.fold2.arff"))),
											 new Instances(new FileReader(new File(args[0]+"badges.fold3.arff"))),
											 new Instances(new FileReader(new File(args[0]+"badges.fold4.arff"))),
											 new Instances(new FileReader(new File(args[0]+"badges.fold5.arff")))};
		Instances train;
		Instances test;
		Id3 classifier;
		Evaluation evaluation;
		Scanner in = new Scanner(System.in);
		
		// The last attribute (index N-1) is the class label
		for(int i = 0 ; i < 5 ; i++)
			datas[i].setClassIndex(datas[i].numAttributes() - 1);
		
		System.out.println();
		System.out.println("Choose which algorithm to run test on");
		System.out.println("---------------------------------------");
		System.out.println("1: Decision Tree");
		System.out.println("2: Decision Trump(max depth 4)");
		System.out.println("3: Decision Trump(max depth 8)");
		System.out.println("4: SGD using LMS");
		System.out.println("5: SGD over Decision Trump(max depth 4)");
		System.out.println("---------------------------------------");
		System.out.println("Your input: ");
		int algorithmChooser = 0;
		while(algorithmChooser<1 || algorithmChooser>5)
		{
			algorithmChooser = in.nextInt();
			if(algorithmChooser > 5 || algorithmChooser<1)
			{
				System.out.println("please provide valid input: ");
			}
		}
		float avgAcc = 0;
		double stdDv = 0;
		double sumDv = 0;
		double[] accArray = {0,0,0,0,0};
		switch(algorithmChooser)
		{
			case 1:
				
				for(int i = 0; i < 5 ; i++)
				{
					train = mergeExcept(datas,i);
					test = datas[i];
					//ID3 classifier for depth selection
					classifier = new Id3();
					classifier.setMaxDepth(-1);
					//Train on train data and display
					classifier.buildClassifier(train);
					//System.out.println(classifier);
					//System.out.println();
					//evaluate test data and display
					evaluation = new Evaluation(test);
					evaluation.evaluateModel(classifier, test);
					avgAcc += evaluation.pctCorrect();
					accArray[i] = evaluation.pctCorrect();
					//System.out.println(evaluation.toSummaryString());
				}
				avgAcc = avgAcc/5;
				
				for(int i = 0; i < 5 ; i++)
				{
					sumDv += ((avgAcc - accArray[i])*(avgAcc - accArray[i]));
				}
				stdDv = Math.sqrt(sumDv/4);
				System.out.println("average Accuracy : "+ avgAcc + "%");
				
				System.out.println("standard Deviation : "+ stdDv + "%");
				break;
			case 2:
				for(int i = 0; i < 5 ; i++)
				{
					train = mergeExcept(datas,i);
					test = datas[i];
					//ID3 classifier for depth selection
					classifier = new Id3();
					classifier.setMaxDepth(4);
					//Train on train data and display
					classifier.buildClassifier(train);
					//System.out.println(classifier);
					//System.out.println();
					//evaluate test data and display
					evaluation = new Evaluation(test);
					evaluation.evaluateModel(classifier, test);
					avgAcc += evaluation.pctCorrect();
					accArray[i] = evaluation.pctCorrect();
					//System.out.println(evaluation.toSummaryString());
				}
				avgAcc = avgAcc/5;
				for(int i = 0; i < 5 ; i++)
				{
					sumDv += ((avgAcc - accArray[i])*(avgAcc - accArray[i]));
				}
				stdDv = Math.sqrt(sumDv/4);
				System.out.println("average Accuracy : "+ avgAcc + "%");
				
				System.out.println("standard Deviation : "+ stdDv + "%");
				break;

			case 3:
				for(int i = 0; i < 5 ; i++)
				{
					train = mergeExcept(datas,i);
					test = datas[i];
					//ID3 classifier for depth selection
					classifier = new Id3();
					classifier.setMaxDepth(8);
					//Train on train data and display
					classifier.buildClassifier(train);
					//System.out.println(classifier);
					//System.out.println();
					//evaluate test data and display
					evaluation = new Evaluation(test);
					evaluation.evaluateModel(classifier, test);
					avgAcc += evaluation.pctCorrect();
					accArray[i] = evaluation.pctCorrect();
					//System.out.println(evaluation.toSummaryString());
				}
				avgAcc = avgAcc/5;
				for(int i = 0; i < 5 ; i++)
				{
					sumDv += ((avgAcc - accArray[i])*(avgAcc - accArray[i]));
				}
				stdDv = Math.sqrt(sumDv/4);
				System.out.println("average Accuracy : "+ avgAcc + "%");
				
				System.out.println("standard Deviation : "+ stdDv + "%");
				break;
			case 4:
				break;
			case 5:
				break;
				
			default:
				return;
		}
		
    }
}

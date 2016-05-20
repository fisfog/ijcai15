#include <stdio.h>

#include <stdlib.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <direct.h>
#include <Windows.h>
#include <assert.h>
using namespace std;




typedef double valueType;

const valueType ftrl_alpha = 0.1f;
const valueType ftrl_beta = 1.0f;
const valueType ftrl_lambda1 = 1.0f;
const valueType ftrl_lambda2 = 1.0f;

static valueType sgn(valueType input)
{
	return input > 0 ? 1.0f : -1.0f;
}

static void ftrlProcess(valueType& W, valueType& g, valueType& q, valueType& z)
{
	valueType sigma = 0.0f;
	sigma = (sqrt(q + g*g) - sqrt(q)) / ftrl_alpha;
	q = q + g*g;
	z = z + g - W*sigma;
	if (abs(z) < ftrl_lambda1)
	{
		W = 0.0f;
	}
	else
	{
		W = -1.0f / (ftrl_lambda2 + (ftrl_beta + sqrt(q)) / ftrl_alpha) * (z - ftrl_lambda1 * sgn(z));
	}
}

valueType safeExp(valueType x)
{
	if (x > 19) x = 19;
	if (x < -19) x = -19;

	return exp(x);
}

valueType safeLog(valueType x)
{
	if (x < 1e-19)
		x = 1e-19;
	return log(x);
}
valueType dotProduct(const vector<valueType>& a, const vector<valueType>& b)
{
	assert(a.size() == b.size());
	valueType result = 0.0;
	for (int i = 0; i < a.size(); i++)
	{
		result += a[i] * b[i];
	}

	return result;

}

struct ftrlStruct
{
	valueType q;
	valueType z;
	ftrlStruct()
	{
		q = 0.0;
		z = 0.0;
	};
};

valueType mFunc(const vector<valueType>& feature, const vector<valueType>& weights)
{
	return dotProduct(feature, weights);
}

valueType sigmoid(valueType x)
{
	return 1.0 / (1.0 + safeExp(x));
}

valueType predict(const vector<valueType>& feature,const vector<valueType>& weights)
{
	return sigmoid(0.0 - mFunc(feature, weights));
}



void trainWithOneFeature(const vector<valueType>& feature, vector<valueType>& weights, vector<ftrlStruct>& ftrlInfo, int label)
{
	assert(label == -1 || label == 1);
	valueType mFuncValue = mFunc(weights, feature);

	valueType y = label > 0 ? 1.0f : -1.0f;
	for (int i = 0; i < feature.size(); i++)
	{
		valueType W = weights[i];
		valueType x = feature[i];
		valueType gradientW = (0.0f - sigmoid(mFuncValue*y)*y*x);
	
		ftrlProcess(weights[i], gradientW,
			ftrlInfo[i].q,
			ftrlInfo[i].z);
	}

}

const int Dimension = 13;//for example

void loadLabel(const string& fileName, vector<int>& label, int number)
{
	label.resize(number, 0);
	FILE* fp = fopen(fileName.c_str(), "r");
	for (int i = 0; i < number; i++)
	{
		fscanf(fp, "%d\n", &label[i]);
	}
	fclose(fp);
}

void loadDataset(const string& fileName, vector<vector<valueType> > & dataset, int number)
{
	FILE* fp = fopen(fileName.c_str(), "r");

	dataset.resize(number, vector<valueType>(Dimension, 0.0));

	for (int i = 0; i < number; i++)
	{
		for (int j = 0; j < Dimension-1; j++)
		{
			fscanf(fp, "%lf,", &dataset[i][j]);
		}
		fscanf(fp, "%lf\n", &dataset[i][Dimension - 1]);
	}


	fclose(fp);
}


valueType pair_predict(const vector<valueType>& feature, const vector<vector<valueType> >& trainDataset,
	const vector<int>& trainLabel, const vector<valueType>& weights)
{
	assert(trainDataset.size()>0);
	valueType result = 0.0;

	assert(trainDataset.size() == trainLabel.size());
	for (int i = 0; i < trainDataset.size(); i++)
	{
		vector<valueType> one_instance;
		one_instance = feature;
		one_instance.insert(one_instance.end(), trainDataset[i].begin(), trainDataset[i].end());
		valueType temResult = predict(one_instance, weights);
		if (trainLabel[i] == 1)
			temResult = 1.0 - temResult;
		
		result += temResult;
	}

	return result / trainDataset.size();
}

int main()
{
	_chdir("C:\\shen\\pairTrain\\");

	int trainNumber=260864;
	int testNumber = 261477;

//	int trainNumber = 100;
//	int testNumber = 100;
	string trainDataName = "result_of_train17.csv";
	string testDataName = "result_of_test17.csv";
	string trainLabelName = "label_of_train.csv";
	string testResultName = "probability.txt";

	vector<vector<valueType> > trainDataset;
	vector<vector<valueType> > testDataset;

	vector<int> trainLabel;


	loadDataset(trainDataName, trainDataset, trainNumber);
	loadDataset(testDataName, testDataset, testNumber);
	loadLabel(trainLabelName, trainLabel,trainNumber);


	assert(trainDataset.size() == trainLabel.size());

	vector<valueType> weights(Dimension*2, 0.0);
	vector<ftrlStruct> ftrlInfo(Dimension*2);

	for (int i = 0; i < trainDataset.size()-1; i++)
	{
		if (i % 20 == 0)
			printf("processing NO. %d: ", i);

		for (int j = i+1; j < trainDataset.size(); j++)
		{
			vector<valueType> one_feature = trainDataset[i];
			one_feature.insert(one_feature.end(), trainDataset[j].begin(), trainDataset[j].end());
			int one_label=1;
			if (trainLabel[i] == trainLabel[j])
				one_label = -1;

			trainWithOneFeature(one_feature, weights, ftrlInfo, one_label);
		}

	}

	FILE* fp = fopen("weights.txt", "w");
	for (int i = 0; i < weights.size(); i++)
	{
		fprintf(fp,"%lf\n", weights[i]);
	}
	fclose(fp);

	fp = fopen(testResultName.c_str(), "w");

	for (int i = 0; i < testDataset.size(); i++)
	{
		if (i % 3000 == 0)
			printf("test No. %d\n", i);
		fprintf(fp, "%lf\n", pair_predict(testDataset[i],trainDataset,trainLabel,weights));
	}

	fclose(fp);

	return 0;
}
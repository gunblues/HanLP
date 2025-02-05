package com.hankcs.hanlp.classification.classifiers;

import com.hankcs.hanlp.classification.statistics.evaluations.FMeasure;
import com.hankcs.hanlp.utility.MathUtility;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.classification.corpus.*;
import com.hankcs.hanlp.classification.features.ChiSquareFeatureExtractor;
import com.hankcs.hanlp.classification.features.BaseFeatureData;
import com.hankcs.hanlp.classification.models.AbstractModel;
import com.hankcs.hanlp.classification.models.NaiveBayesModel;

import static com.hankcs.hanlp.classification.utilities.io.ConsoleLogger.logger;

import java.util.*;
import java.io.PrintWriter;

/**
 * 实现一个基于多项式贝叶斯模型的文本分类器
 */
public class NaiveBayesClassifier extends AbstractClassifier
{

    private NaiveBayesModel model;

    /**
     * 由训练结果构造一个贝叶斯分类器，通常用来加载磁盘中的分类器
     *
     * @param naiveBayesModel
     */
    public NaiveBayesClassifier(NaiveBayesModel naiveBayesModel)
    {
        this.model = naiveBayesModel;
    }

    /**
     * 构造一个空白的贝叶斯分类器，通常准备用来进行训练
     */
    public NaiveBayesClassifier()
    {
        this(null);
    }

    /**
     * 获取训练结果
     *
     * @return
     */
    public NaiveBayesModel getNaiveBayesModel()
    {
        return model;
    }

    public void train(IDataSet dataSet)
    {
        logger.out("原始数据集大小:%d\n", dataSet.size());
        //选择最佳特征
        BaseFeatureData featureData = selectFeatures(dataSet);

        //初始化分类器所用的数据
        model = new NaiveBayesModel();
        model.n = featureData.n; //样本数量
        model.d = featureData.featureCategoryJointCount.length; //特征数量

        model.c = featureData.categoryCounts.length; //类目数量
        model.logPriors = new TreeMap<Integer, Double>();

        int sumCategory;
        for (int category = 0; category < featureData.categoryCounts.length; category++)
        {
            sumCategory = featureData.categoryCounts[category];
            model.logPriors.put(category, Math.log((double) sumCategory / model.n));
        }

        //拉普拉斯平滑处理（又称加一平滑），这时需要估计每个类目下的实例
        Map<Integer, Double> featureOccurrencesInCategory = new TreeMap<Integer, Double>();

        Double featureOccSum;
        for (Integer category : model.logPriors.keySet())
        {
            featureOccSum = 0.0;
            for (int feature = 0; feature < featureData.featureCategoryJointCount.length; feature++)
            {

                featureOccSum += featureData.featureCategoryJointCount[feature][category];
            }
            featureOccurrencesInCategory.put(category, featureOccSum);
        }

        //对数似然估计
        int count;
        int[] featureCategoryCounts;
        double logLikelihood;
        for (Integer category : model.logPriors.keySet())
        {
            for (int feature = 0; feature < featureData.featureCategoryJointCount.length; feature++)
            {

                featureCategoryCounts = featureData.featureCategoryJointCount[feature];

                count = featureCategoryCounts[category];

                logLikelihood = Math.log((count + 1.0) / (featureOccurrencesInCategory.get(category) + model.d));
                if (!model.logLikelihoods.containsKey(feature))
                {
                    model.logLikelihoods.put(feature, new TreeMap<Integer, Double>());
                }
                model.logLikelihoods.get(feature).put(category, logLikelihood);
            }
        }
        logger.out("贝叶斯统计结束\n");
        model.catalog = dataSet.getCatalog().toArray();
        model.tokenizer = dataSet.getTokenizer();
        model.wordIdTrie = featureData.wordIdTrie;
    }

    public AbstractModel getModel()
    {
        return model;
    }


    public Map<String, Double> predict(String text) throws IllegalArgumentException, IllegalStateException
    {
        if (model == null)
        {
            throw new IllegalStateException("未训练模型！无法执行预测！");
        }
        if (text == null)
        {
            throw new IllegalArgumentException("参数 text == null");
        }

        //分词，创建文档
        Document doc = new Document(model.wordIdTrie, model.tokenizer.segment(text));

        return predict(doc);
    }


    public List<String> predict(String text, int count) throws IllegalArgumentException, IllegalStateException
    {
        if (model == null)
        {
            throw new IllegalStateException("未训练模型！无法执行预测！");
        }
        if (text == null)
        {
            throw new IllegalArgumentException("参数 text == null");
        }

        //分词，创建文档
        Document doc = new Document(model.wordIdTrie, model.tokenizer.segment(text));

        double minScore = 0.01;
        Map<Double, String> scoreMap = predict(doc, minScore);

        List<String> categories = new ArrayList<String>();
        for(Map.Entry<Double,String> entry : scoreMap.entrySet()) {
            Double score = entry.getKey();
            String category = entry.getValue();
            if (categories.size() < count) {
                categories.add(String.format("%s\t%f", category, score));
            }
        }

        return categories;
    }

    @Override
    public double[] categorize(Document document) throws IllegalArgumentException, IllegalStateException
    {
        Integer category;
        Integer feature;
        Integer occurrences;
        Double logprob;

        double[] predictionScores = new double[model.catalog.length];
        long startTime = System.currentTimeMillis();
        for (Map.Entry<Integer, Double> entry1 : model.logPriors.entrySet())
        {
            category = entry1.getKey();
            logprob = entry1.getValue(); //用类目的对数似然初始化概率

            //对文档中的每个特征
            for (Map.Entry<Integer, int[]> entry2 : document.tfMap.entrySet())
            {
                feature = entry2.getKey();

                if (!model.logLikelihoods.containsKey(feature))
                {
                    continue; //如果在模型中找不到就跳过了
                }

                occurrences = entry2.getValue()[0]; //获取其在文档中的频次

                logprob += occurrences * model.logLikelihoods.get(feature).get(category); //将对数似然乘上频次
            }
            predictionScores[category] = logprob;
        }

        if (configProbabilityEnabled) {
            MathUtility.normalizeExp(predictionScores);
        }

        return predictionScores;
    }

    /**
     * 统计特征并且执行特征选择，返回一个FeatureStats对象，用于计算模型中的概率
     *
     * @param dataSet
     * @return
     */
    protected BaseFeatureData selectFeatures(IDataSet dataSet)
    {
        ChiSquareFeatureExtractor chiSquareFeatureExtractor = new ChiSquareFeatureExtractor();

        logger.start("使用卡方检测选择特征中...");
        //FeatureStats对象包含文档中所有特征及其统计信息
        BaseFeatureData featureData = chiSquareFeatureExtractor.extractBasicFeatureData(dataSet); //执行统计

        //我们传入这些统计信息到特征选择算法中，得到特征与其分值
        Map<Integer, Double> selectedFeatures = chiSquareFeatureExtractor.chi_square(featureData);

        //从统计数据中删掉无用的特征并重建特征映射表
        int[][] featureCategoryJointCount = new int[selectedFeatures.size()][];
        featureData.wordIdTrie = new BinTrie<Integer>();
        String[] wordIdArray = dataSet.getLexicon().getWordIdArray();
        Map<String, ArrayList<String>> categoryFeatures = new HashMap<>();
        String[] catalog = dataSet.getCatalog().toArray();
        int p = -1;
        for (Integer feature : selectedFeatures.keySet())
        {
            featureCategoryJointCount[++p] = featureData.featureCategoryJointCount[feature];
            featureData.wordIdTrie.put(wordIdArray[feature], p);

            for (int category = 0;  category < featureData.featureCategoryJointCount[feature].length; category++) {
                int featureCount = featureData.featureCategoryJointCount[feature][category];
                if (featureCount > 0) {
                    ArrayList<String> featureList = null;
                    if (!categoryFeatures.containsKey(catalog[category])) {
                        featureList = new ArrayList<String>();
                        featureList.add(wordIdArray[feature]);
                    } else {
                        featureList = categoryFeatures.get(catalog[category]);

                        if (!featureList.contains(wordIdArray[feature])) {
                            featureList.add(wordIdArray[feature]);
                        }
                    }
                    categoryFeatures.put(catalog[category], featureList);
                }
            }
        }
        logger.finish(",选中特征数:%d / %d = %.2f%%\n", featureCategoryJointCount.length,
                      featureData.featureCategoryJointCount.length,
                      featureCategoryJointCount.length / (double)featureData.featureCategoryJointCount.length * 100.);
        featureData.featureCategoryJointCount = featureCategoryJointCount;

        Arrays.sort(wordIdArray);
        try {
            PrintWriter writer = new PrintWriter("/tmp/selected_features.txt", "UTF-8");
            for (String word : wordIdArray) {
                writer.println(word);
            }
            writer.close();

            writer = new PrintWriter("/tmp/selected_features_by_category.txt", "UTF-8");
            for(Map.Entry<String, ArrayList<String>> entry : categoryFeatures.entrySet()) {
                String key = entry.getKey();
                ArrayList<String> value = entry.getValue();
                writer.printf("%s : \n%s\n\n", key, String.join("\n", value));
            }
            writer.close();

        } catch (Exception e) {}

        return featureData;
    }

    public FMeasure evaluate(IDataSet testingDataSet, double cutOffPoint)
    {
        int c = this.model.catalog.length;
        double[] TP_FP = new double[c]; // 判定为某个类别的数量
        double[] TP_FN = new double[c]; // 某个类别的样本数量
        double[] TP = new double[c];    // 判定为某个类别且判断正确的数量
        String[] catalogs = testingDataSet.getCatalog().toArray();
        double time = System.currentTimeMillis();

        System.out.println("**************** guess wrong ***************************");
        System.out.println("predict\tactual\tproduct_title\tscore");
        for (Document document : testingDataSet)
        {
            final Map<Integer, Double> scoreMap = this.label(document);
            Map.Entry<Integer,Double> entry = scoreMap.entrySet().iterator().next();
            Integer out = entry.getKey();
            Double score = entry.getValue();
            final int key = document.category;

            if (score >= cutOffPoint) {
                ++TP_FP[out];
            }

            ++TP_FN[key];

            if (key == out && score >= cutOffPoint)
            {
                ++TP[out];
            }

            if (key != out && score >= cutOffPoint) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < document.tokenArray.length; i++) {
                    sb.append(document.tokenArray[i]);
                    if (i != document.tokenArray.length - 1) {
                        sb.append(" ");
                    }
                }
                String content = sb.toString();
                String message = String.format("%s\t%s\t%s\t%f", catalogs[out], catalogs[key], content, score);
                System.out.println(message);
            }
        }
        System.out.println("********************************************************");
        time = System.currentTimeMillis() - time;

        FMeasure result = calculate(c, testingDataSet.size(), TP, TP_FP, TP_FN);
        result.catalog = testingDataSet.getCatalog().toArray();
        result.speed = result.size / (time / 1000.);

        return result;
    }

    /**
     *
     * @param c 类目数量
     * @param size 样本数量
     * @param TP 判定为某个类别且判断正确的数量
     * @param TP_FP 判定为某个类别的数量
     * @param TP_FN 某个类别的样本数量
     * @return
     */
    private static FMeasure calculate(int c, int size, double[] TP, double[] TP_FP, double[] TP_FN)
    {
        double precision[] = new double[c];
        double recall[] = new double[c];
        double f1[] = new double[c];
        double accuracy[] = new double[c];
        FMeasure result = new FMeasure();
        result.size = size;

        for (int i = 0; i < c; i++)
        {
            double TN = result.size - TP_FP[i] - (TP_FN[i] - TP[i]);
            accuracy[i] = (TP[i] + TN) / result.size;
            if (TP[i] != 0)
            {
                precision[i] = TP[i] / TP_FP[i];
                recall[i] = TP[i] / TP_FN[i];
                result.average_accuracy += TP[i];
            }
            else
            {
                precision[i] = 0;
                recall[i] = 0;
            }
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]);
        }
        result.average_precision = MathUtility.average(precision);
        result.average_recall = MathUtility.average(recall);
        result.average_f1 = 2 * result.average_precision * result.average_recall
            / (result.average_precision + result.average_recall);
        result.average_accuracy /= (double) result.size;
        result.accuracy = accuracy;
        result.precision = precision;
        result.recall = recall;
        result.f1 = f1;
        return result;
    }
}

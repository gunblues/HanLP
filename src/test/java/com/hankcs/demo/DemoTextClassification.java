/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/20 AM11:46</create-date>
 *
 * <copyright file="DemoAtFirstSight.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.demo;


import com.hankcs.hanlp.classification.classifiers.IClassifier;
import com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier;
import com.hankcs.hanlp.classification.corpus.FileDataSet;
import com.hankcs.hanlp.classification.corpus.IDataSet;
import com.hankcs.hanlp.classification.corpus.MemoryDataSet;
import com.hankcs.hanlp.classification.models.NaiveBayesModel;
import com.hankcs.hanlp.classification.statistics.evaluations.Evaluator;
import com.hankcs.hanlp.classification.statistics.evaluations.FMeasure;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.TestUtility;
import com.hankcs.hanlp.classification.tokenizers.BlankTokenizer;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * 第一个demo,演示文本分类最基本的调用方式
 *
 * @author hankcs
 */
public class DemoTextClassification
{
    /**
     * 搜狗文本分类语料库5个类目，每个类目下1000篇文章，共计5000篇文章
     */
    public static final String CORPUS_FOLDER = "data/test/mini_orca_training_data"; // TestUtility.ensureTestData("搜狗文本分类语料库迷你版", "http://hanlp.linrunsoft.com/release/corpus/sogou-text-classification-corpus-mini.zip");

    /**
     * 模型保存路径
     */
    public static final String MODEL_PATH = "data/test/classification-mini-orca-model.ser";


    public static void main(String[] args) throws IOException
    {
        IClassifier classifier = new NaiveBayesClassifier(trainOrLoadModel());
/*        predict(classifier, "C罗获2018环球足球奖最佳球员 德尚荣膺最佳教练");
        predict(classifier, "英国造航母耗时8年仍未服役 被中国速度远远甩在身后");
        predict(classifier, "研究生考录模式亟待进一步专业化");
        predict(classifier, "如果真想用食物解压,建议可以食用燕麦");
        predict(classifier, "通用及其部分竞争对手目前正在考虑解决库存问题");*/
        predict(classifier, "aife life iPhone htc 智慧型 手機 水鑽 鑽石 耳機孔 防塵塞 耳機塞 防潮塞 歡迎 大量 批發");
        predict(classifier, "EF 橫線 點 點 襪 - 灰 22 24 cm愛買");
        predict(classifier, "德國WMF Perfect Plus 22公分 快易鍋 (4.5L)");
        predict(classifier, "ASUS 華碩 Full HD 低藍光不閃屏螢幕 - 22型 (VP229DA)");
        predict(classifier, "acer 宏碁 va 面板 4k 解析度 液晶 螢幕 32 型 et322qk");
    }

    private static void predict(IClassifier classifier, String text)
    {
        /*
        for(Map.Entry<String, Double> entry : classifier.predictV2(text).entrySet()) {
            System.out.printf("《%s》 属于分类 【%s】, 【%f】\n", text, entry.getKey(), entry.getValue());
        }
        */

        List<String> categories = classifier.predict(text, 3);
        for (String category : categories) {
            System.out.printf("《%s》 属于分类 【%s】\n", text, category);
        }

        // System.out.printf("《%s》 属于分类 【%s】\n", text, classifier.classify(text));
    }

    private static NaiveBayesModel trainOrLoadModel() throws IOException
    {
        NaiveBayesModel model = (NaiveBayesModel) IOUtil.readObjectFrom(MODEL_PATH);
        if (model != null)
        {
            return model;
        }

        File corpusFolder = new File(CORPUS_FOLDER);
        if (!corpusFolder.exists() || !corpusFolder.isDirectory())
        {
            System.err.println("没有文本分类语料，请阅读IClassifier.train(java.lang.String)中定义的语料格式与语料下载：" +
                                   "https://github.com/hankcs/HanLP/wiki/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E4%B8%8E%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90");
            System.exit(1);
        }

        IClassifier classifier = new NaiveBayesClassifier(); // 创建分类器，更高级的功能请参考IClassifier的接口定义

        IDataSet trainingCorpus = new FileDataSet().                          // FileDataSet省内存，可加载大规模数据集
            setBlankTokenizer().                               // 支持不同的ITokenizer，详见源码中的文档
            load(CORPUS_FOLDER, "UTF-8", 0.9);
        classifier.train(trainingCorpus);

        model = (NaiveBayesModel) classifier.getModel();

        IDataSet testingCorpus = new MemoryDataSet(model).
            load(CORPUS_FOLDER, "UTF-8", -0.1);        // 后10%作为测试集
        // 计算准确率
        FMeasure result = ((NaiveBayesClassifier)classifier).evaluate(testingCorpus,0.9);
        System.out.println(result);


        IOUtil.saveObjectTo(model, MODEL_PATH);
        return model;
    }
}

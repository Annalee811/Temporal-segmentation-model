# 同图同V35掩膜的V90时序流程

V35是唯一的根系/背景来源。输入为已经匹配的66张500×500原RGB和用户选定的V35预测mask。V90只预测现有根系像素的新生/既有类别。不执行配准、裁图、视野提取或缩放。

把本模型包与V90_savedV35_inputs.zip解压至同一目录，然后：

```bash
python -m pip install -r requirements.txt
python 04_ConvLSTM_GAT_V90/predict_saved_v35.py --masks-zip inputs/V35_selected_predictions.zip --parents inputs/parents.csv --RGB-root inputs/RGB --runtime runtime --output results
```

V35已有mask是不可改变的前景。为了V66软特征，按同一500px输入、同一第91轮V35恢复概率并断言其0.5阈值逐像素复现mask；V90阶段使用缓存的概率，只运行时序先验、ConvLSTM/GAT、Axis、Junction。任何根系前景差异都会停止输出。

原V35预测ZIP包含原提交标记文件，但本程序只读root_mask与root_overlay，不读取submitted_mask。缺期不生成图像；T1可见根作为既有根参考。延伸/持续合并为既有，输出0背景、1既有、2新生。

三份模型权重与之前一致，没有额外训练。现有V90迁移接口的骨架分区/邻接和一个交叉度数代理仍不同于原张量绑定预处理，原未发布稠密局部修正预处理未包括；差异保留于provenance。

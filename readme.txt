问题描述：
已知某地的从1990年到2009年的人口数量，机动车数量，公路面积，公路客运量以及公路货运量五个参数，基于历史数据构建预测模型（前三参数预测后二）

解决方法：
传统BP神经网络搭建预测模型

三个程序的不同：
bp_matlab_new，基于新版本newff函数搭建网络，新版本速度快，但准确度不高（原因是新版本将训练样本以6:2:2比例分为训练集，验证集和测试集，训练样本减少）
bp_matlab_new，基于旧版本newff函数搭建网络，旧版本速度慢，但准确度很高
bp_myself            自己写的BP算法的预测程序，速度快，准确度高，兼有前面两程序优点，而且网络参数可方便调整

参考资料：
1.matlab newff新旧版本详解：
https://blog.csdn.net/guyuealian/article/details/66969232?utm_source=itdadao&utm_medium=referral
http://blog.sina.com.cn/s/blog_64b046c70101cko4.html

2.传统BP算法的推导（很nice很清楚）：
http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B

3.编程技巧学习：
《MATLAB在数学建模中的应用》

附：
这是本人第一个不当调包侠写出来的有关神经网络的程序，虽然编写的时候已经考虑到了程序的扩展性，可读性问题，基本上每句的意义也都注释了出来，
但应该还是有些冗余之处。
练习编程与加深算法理解为主，学习交流为辅，欢迎各位大佬斧正！
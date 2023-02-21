## hw1

主要难点在几个算子的后向的实现

* reshape & transpose: 原样转回去  
* broadcast_to & summation: 原理也是转回去, 不过稍微复杂一些
    * broadcast的原理是从右往左匹配, 不相等就广播, 找到广播的dim做reduce_sum
    * summation的原理刚好相反, 找到减掉的那几个dim广播回去
* matmul: 上一个hw学习了矩阵乘法的求导, 但是这里还要注意广播, 如果存在也要reduce_sum回去

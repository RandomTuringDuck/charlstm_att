
损失函数计算的是每个字符的交叉熵损失，也可以以句为单位，那就乘上句长，但有个问题是，因为用的是batch，所以很多句子的长度不等，
用<pad>来填充的，所以可能会有误差。

写的匆忙，没有讲训练函数和验证函数统一整合到一起。

数据集的质量也决定了训练模型的质量，仅仅处理了过短的句子，就bleu提高了


> nltk.translate.bleu_score.corpus_bleu

模型|bleu1|bleu2|bleu3|bleu4
---|---|---|---|---
lstm_twin|0.5105|0.3450|0.2523|0.1959
lstm|0.4959|0.3315|0.2397|0.1835
attention|0.4859|0.3206|0.2283|0.1718


lstm|lstm_twin|lstm_attention
---|---|--- 
(什么也不想说了)，下次还会再来的|什么也不想说了，我也是醉了，不过还是很喜欢的|(什么也不想说了)，不过这个价格也就这样了  
(做工很精细)，款式很漂亮，很喜欢  |(做工很精细)，款式很漂亮，做工精致，很满意|（做工很精细），款式很漂亮，做工精致，很满意  
(比我预料中的好)，很喜欢，卖家服务态度很好，物流也很快，很满意的一次购物  |比我预料中的好多了，很喜欢，下次还会再来的|（比我预料中的好），一直都在京东买，值得信赖  
(这个商品)，很满意的一次购物  |这个商品还不错，就是有点小贵，不过还是很满意的|（这个商品是正品），质量好，价格实惠，值得购买  
(一直没收到)，很满意的一次购物  |一直没收到，不过还没有用，不知道效果怎么样|（一直没收到），但是不知道是不是正品，但是不知道是不是正品  
(外观很漂亮)，很满意  |外观漂亮，做工精致，款式漂亮，价格实惠，很满意| （外观漂亮），质量也不错，很喜欢，下次还会光顾  
(这款游戏)很好，值得购买 |这款游戏很好用，不过还没有用，不知道效果怎么样| （这款游戏不错），就是有点小，不过还是很满意的 
(想要的东西)，很满意  |想要的一本书，看看效果，看着还不错| （想要的东西），不过还是很满意的，就是不知道怎么样，不过还是很满意的  
(书)很好，很喜欢  |书很好，很喜欢，看着很好看，很喜欢| （书）很好，送货速度快，快递员服务态度好，快递员服务态度好，给好评  
(店家)服务态度很好，很满意的一次购物 |店家服务态度很好，物流也很快，很满意的一次购物| （店家）服务态度很好，发货很快，包装很好，做工精致，很满意
(服务)态度也很好，下次还会再来的  |服务态度好，送货速度快，服务态度好| （服务）态度很好，发货速度很快，包装很好，包装很好，很满意
(刚收到货)，下次还会再来的  |刚收到，还没用，看着还不错，就是不知道用起来怎么样| （刚收到货），还没用，看着还不错，不知道效果怎么样
(家里人)很喜欢  |家里人都喜欢吃，一直在京东买东西，很好的产品。| (家里人都说好），就是不知道怎么样，不过还是不错的
(有用过这个牌子)，很好用，下次还会再来的 |有用过这个牌子，很好用，很喜欢| (有用过这个)牌子的，感觉还不错，就是不知道是不是正品
(箱子外观很漂亮)，很喜欢  |箱子外观很漂亮，质量很好，很满意| (箱子外观很漂亮)，质量也很好，很满意
(篮球)手感很好，下次还会再来的|篮球手感不错，就是不知道耐不耐用|(篮球)手感不错，就是有点滑，不过还是很满意的






















 


  
  
 
  


# 
-picture, test_picture : 原始圖片放置在此資料夾，以供後續訓練使用
trainpic, testpic : 經處理過的圖片將儲存於此
pridict_pic : 儲存預測用的圖片
t_num.json, d_num.json : 儲存原始資料有多少張 (t_num為test，d_num為train)
train.pickle, test.pickle : 由prepicture.py產生，"data"儲存亂序正規化的訓練資料，"label"儲存對應的標籤
train.py : 訓練並儲存模型 
predict.py : 將predict資料夾內的圖片做辨識
prepicture.py: 圖片前處理及儲存成pickle
show.py : 評估模型
killqueen.h5 : 模型，包含架構與權重

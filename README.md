# 
-picture, test_picture : 原始圖片放置在此資料夾，以供後續訓練使用\n
trainpic, testpic : 經處理過的圖片將儲存於此\n
pridict_pic : 儲存預測用的圖片\n
t_num.json, d_num.json : 儲存原始資料有多少張 (t_num為test，d_num為train)\n
train.pickle, test.pickle : 由prepicture.py產生，"data"儲存亂序正規化的訓練資料，"label"儲存對應的標籤\n
train.py : 訓練並儲存模型 \n
predict.py : 將predict資料夾內的圖片做辨識\n
prepicture.py: 圖片前處理及儲存成pickle\n
show.py : 評估模型\n
killqueen.h5 : 模型，包含架構與權重\n

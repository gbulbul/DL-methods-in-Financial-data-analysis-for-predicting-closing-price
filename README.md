This program aims to compare 3 different DL models, including MLP, CNN and LSTM, with respect to how they did on predicting closing price of 3 different stocks, namely, MSFT, AMZN, NVDIA by using RMSE values.

MSFT model graphs given in order (Top: MLP, Middle: CNN, Bottom: LSTM)
![image](https://github.com/user-attachments/assets/68d14bc1-e6d8-49dc-84f8-d48898b79ce2) ![image](https://github.com/user-attachments/assets/72915433-9ce7-47e6-8921-318f1d2091ec) ![image](https://github.com/user-attachments/assets/ed276c36-873b-45b9-a352-10c6d4dc82a8)


AMZN model graphs given in order (Top: MLP, Middle: CNN, Bottom: LSTM)
![image](https://github.com/user-attachments/assets/04539744-7cd2-4373-8e6f-5da6e725eb5f) ![image](https://github.com/user-attachments/assets/c9795ac7-654e-4c41-b209-be59d6c37e75) ![image](https://github.com/user-attachments/assets/c5092cfe-9b77-43ef-bbdf-632e0c7d11ad)


NVDIA model graphs given in order (Top: MLP, Middle: CNN, Bottom: LSTM)
![image](https://github.com/user-attachments/assets/5df7389e-6001-4fed-8004-f765ae04cc55) ![image](https://github.com/user-attachments/assets/f11a97f7-d26a-463c-92d6-4b0a60224868) ![image](https://github.com/user-attachments/assets/ea711e5a-e0a4-4960-a72c-c2e8576cfcb6)


RMSE values, repsectively for MSFT, AMZN & NVDIA, are listed into 3 categories in terms of modeling method.

LSTM: 4.684305703518117, 288.69500342174814, 43.479115340224716

CNN: 28.9759303984019, 206.0191577680697, 30.381648658707622

MLP: 14.010727210651071, 223.25005226945635, 93.45077633716829

Interpretation of the outputs:

1)In terms of predicting MSFT stock, LSTM did the best job by looking at the lowest RMSE score which is associated with the LSTM.
2)LSTM on MSFT stock obtained the lowest RMSE score, meaning that LSTM is the most suitable model to model closing price for MSFT.
3) Overall, all models got their lowest RMSE values on MSFT stock, saying that MSFT is being able to be predicted very well compared with the other two stocks.
4)CNN modeling looks suitable for NVDIA and AMZN because they had their lowest RMSE values. 

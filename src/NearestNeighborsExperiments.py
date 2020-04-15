import matplotlib.pyplot as plt
from src import GetUserClimbData as gud
from src import UserClimbCollaborativeFiltering as cf

# Colorado data
print('Getting Data')
path = '../user_ids/UserIDsBoulder.csv'
user_rating_df_with_id = gud.get_users_climbs(path)
user_rating_df = user_rating_df_with_id.drop(labels='user_id', axis=1)
user_rating_np = user_rating_df.to_numpy()


# Calculate and plot MAE and RMSE for different values of k
mae_per_k = []
mse_per_k = []
for i in range(5, 30):
    print('Calculating MAE and MSE for k = ' + str(i))
    mae, mse = cf.eval_five_fold(5, user_rating_df)
    mae_per_k.append(mae)
    mse_per_k.append(mse)

plt.plot(range(5, 30), mae_per_k, 'k', label='MAE', linewidth=2)
plt.plot(range(5, 30), mse_per_k, 'r', label='RMSE', linewidth=2)
plt.legend()
plt.xlabel('Value of k for Nearest Neighbors')
plt.ylabel('MAE and RMSE')
plt.show()
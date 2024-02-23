using System;

public class CollaborativeFiltering
{
    private int numUsers;
    private int numItems;
    private int latentFactors;
    private double learningRate;
    private double regParam;
    private int epochs;
    private double[,] userLatent;
    private double[,] itemLatent;
    private Random rand;

    public CollaborativeFiltering(int numUsers, int numItems, int latentFactors = 10, double learningRate = 0.001, double regParam = 0.02, int epochs = 10)
    {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.latentFactors = latentFactors;
        this.learningRate = learningRate;
        this.regParam = regParam;
        this.epochs = epochs;
        this.userLatent = new double[numUsers, latentFactors];
        this.itemLatent = new double[numItems, latentFactors];
        this.rand = new Random();

        // Initialize user and item latent matrices randomly
        for (int i = 0; i < numUsers; i++)
        {
            for (int j = 0; j < latentFactors; j++)
            {
                userLatent[i, j] = rand.NextDouble() * 0.1;
            }
        }

        for (int i = 0; i < numItems; i++)
        {
            for (int j = 0; j < latentFactors; j++)
            {
                itemLatent[i, j] = rand.NextDouble() * 0.1;
            }
        }
    }

    public void Fit(int[,] ratings)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < ratings.GetLength(0); i++)
            {
                int userId = ratings[i, 0];
                int itemId = ratings[i, 1];
                double rating = ratings[i, 2];
                double predictedRating = 0;

                for (int k = 0; k < latentFactors; k++)
                {
                    predictedRating += userLatent[userId, k] * itemLatent[itemId, k];
                }

                double error = rating - predictedRating;

                // Update user and item latent matrices using gradient descent
                for (int k = 0; k < latentFactors; k++)
                {
                    userLatent[userId, k] += learningRate * (error * itemLatent[itemId, k] - regParam * userLatent[userId, k]);
                    itemLatent[itemId, k] += learningRate * (error * userLatent[userId, k] - regParam * itemLatent[itemId, k]);
                }
            }
        }
    }

    public double Predict(int userId, int itemId)
    {
        double predictedRating = 0;

        for (int k = 0; k < latentFactors; k++)
        {
            predictedRating += userLatent[userId, k] * itemLatent[itemId, k];
        }

        return predictedRating;
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        // Toy dataset (user_id, item_id, rating)
        int[,] ratings = new int[,]
        {
            {0, 0, 5},
            {0, 1, 4},
            {1, 0, 3},
            {1, 1, 2},
            {2, 0, 4},
            {2, 1, 1}
        };

        int numUsers = 3;
        int numItems = 2;

        // Initialize and fit collaborative filtering model
        CollaborativeFiltering model = new CollaborativeFiltering(numUsers, numItems);
        model.Fit(ratings);

        // Predict ratings for a user-item pair
        int userId = 0;
        int itemId = 1;
        double predictedRating = model.Predict(userId, itemId);
        Console.WriteLine($"Predicted rating for user {userId} and item {itemId}: {predictedRating}");
    }
}

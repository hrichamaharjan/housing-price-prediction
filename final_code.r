# ---------------------------------------------------
# HOUSING PRICE PREDICTION PROJECT
# Structure: Load Data -> Clean Data -> EDA -> Split -> Model Selection -> Diagnostics -> Fixes -> Final Evaluation
# ---------------------------------------------------

# STEP 1: Load Required Libraries
packages <- c("ggplot2", "dplyr", "corrplot", "caret", "lmtest", "car", "MASS", "olsrr", "glmnet", "GGally")
installed_packages <- packages %in% rownames(installed.packages())
if (any(!installed_packages)) install.packages(packages[!installed_packages])
lapply(packages, library, character.only = TRUE)

# STEP 2: Load and Prepare the Dataset
housing <- read.csv("C:/Users/user/Documents/Linear_final_project/Housing.csv", header = TRUE)

# Convert categorical variables to factor type
categorical_vars <- c("mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus")
housing[categorical_vars] <- lapply(housing[categorical_vars], as.factor)

# Check for missing values
colSums(is.na(housing))

# STEP 3: Exploratory Data Analysis (EDA)
# Scatter plot matrix for important numerical variables
ggpairs(housing[, c("price", "area", "bedrooms", "bathrooms", "stories", "parking")])
# Boxplot: Furnishing Status vs Price
ggplot(housing, aes(x = furnishingstatus, y = price)) +
  geom_boxplot() +
  ggtitle("Price by Furnishing Status")

# Correlation matrix for numeric variables
numeric_vars <- housing[sapply(housing, is.numeric)]
corrplot(cor(numeric_vars), method = "number", type = "upper")

# STEP 4: Data Splitting (Train-Test Split)
set.seed(123)
train_index <- createDataPartition(housing$price, p = 0.8, list = FALSE)
train_data <- housing[train_index, ]
test_data <- housing[-train_index, ]

# STEP 5: Model Selection
# Full model with interactions and quadratic term for area
full_model_housing <- lm(price ~ (area + bedrooms + bathrooms + stories + mainroad + guestroom + 
                                    basement + hotwaterheating + airconditioning + parking + 
                                    prefarea + furnishingstatus) , 
                         data = train_data)

# Summary of the full model
summary(full_model_housing)

# Backward Stepwise Selection
backward_model_housing <- step(full_model_housing, direction = "backward")

# Summary of the selected model
summary(backward_model_housing)

# 7. Diagnostics for Initial Model

plot(backward_model_housing, which = 1)
plot(backward_model_housing, which = 2)

# Get residuals
residuals <- resid(backward_model_housing)

# Plot histogram
hist(residuals, 
     breaks = 20,    # number of bins
     col = "lightblue", 
     main = "Histogram of Residuals", 
     xlab = "Residuals")



# STEP 6: Multicollinearity Check (VIF)
vif_values_housing <- vif(backward_model_housing, type = "predictor")
print(vif_values_housing)
vif(backward_model_housing)

# STEP 7: Outlier and Influential Case Detection
# Hat Values
hat_values_housing <- hatvalues(backward_model_housing)
outlier_threshold_housing <- 2 * (length(coef(backward_model_housing)) / nrow(train_data))
print(outlier_threshold_housing)
outliers <- which(hat_values_housing > outlier_threshold_housing)
print(outliers)
# Display outliers with hat values
cat("Outlier Threshold (Hat Values):", outlier_threshold_housing, "\n")
if (length(outliers) > 0) {
  cat("Potential Outliers based on Hat Matrix:\n")
  print(data.frame(Observation = outliers, Hat_Value = hat_values_housing[outliers]))
} else {
  cat("No potential outliers based on Hat Matrix.\n")
}


# Detect and remove influential points (Before cleaning)
cooks_d_housing <- cooks.distance(backward_model_housing)

influential_threshold_housing <- 4 / nrow(train_data)
print(influential_threshold_housing)

# Identify Outliers and Influential Cases

influential_cases_housing <- which(cooks_d_housing > influential_threshold_housing)
# Display influential cases with Cook's Distance values
cat("Influential Threshold (Cook's Distance):", influential_threshold_housing, "\n")
if (length(influential_cases_housing) > 0) {
  cat("Influential Cases based on Cook's Distance:\n")
  print(data.frame(Observation = influential_cases_housing, Cook_Distance = cooks_d[influential_cases_housing]))
} else {
  cat("No influential cases based on Cook's Distance.\n")
}
# Plot Diagnostic Graphs
par(mfrow = c(2, 2))
plot(backward_model_housing)

# Plot Hat Values
plot(hat_values_housing, main = "Hat Values", ylab = "Hat Values", xlab = "Observations")
abline(h = outlier_threshold_housing, col = "red", lty = 2)

# Plot Cook's Distance
plot(cooks_d_housing, main = "Cook's Distance", ylab = "Cook's Distance", xlab = "Observations")
abline(h = influential_threshold_housing, col = "blue", lty = 2)

# STEP 8: Data Cleaning
# Combine outliers and influential observations
influential_outliers <- c(5, 59,101)  # Observations identified as both outliers and influential cases
train_data_cleaned <- train_data[!(rownames(train_data) %in% influential_outliers), ]
total_influential_housing <- union(outliers_housing, influential_cases_housing)

# Remove these observations from training data
train_data_cleaned <- train_data[-total_influential_housing, ]

# STEP 9: Initial Model Diagnostics
# Refit model on cleaned training data
final_housing_model <- lm(
  price ~ area + bedrooms + bathrooms + stories + mainroad + guestroom + 
    basement + hotwaterheating + airconditioning + parking + prefarea + furnishingstatus +
    area:bedrooms + area:bathrooms + area:stories + area:parking +  # Only meaningful interactions
    bedrooms:bathrooms + bathrooms:stories +
    furnishingstatus:area + furnishingstatus:stories,
  data = train_data_cleaned
)

summary(final_housing_model)

names(train_data_cleaned)

# Remove influential observations from the test data
test_data_cleaned <- test_data[!(rownames(test_data) %in% influential_outliers), ]

# Remove variables from the test data that were removed from the training data
# Refitting the model on the cleaned training data

summary(final_housing_model)

par(mfrow = c(2, 2))
plot(final_housing_model)
# Histogram of Residuals
hist(resid(final_housing_model), breaks = 20, col = "lightblue", main = "Histogram of Residuals ")

# Ensure the test data has the same structure as the cleaned training data
if (!all(names(test_data_cleaned) == names(train_data_cleaned))) {
  # Add missing columns to the test data (if any) and fill with NA
  missing_columns <- setdiff(names(train_data_cleaned), names(test_data_cleaned))
  for (col in missing_columns) {
    test_data_cleaned[[col]] <- NA
  }
  
  # Ensure columns are in the same order as the cleaned training data
  test_data_cleaned <- test_data_cleaned[, names(train_data_cleaned)]
}

# Check if there are factor variables in the model
factor_columns <- sapply(train_data_cleaned, is.factor)

# For each factor column in the training data, ensure the same factor levels in the test data
for (col in names(test_data_cleaned)[factor_columns]) {
  test_data_cleaned[[col]] <- factor(test_data_cleaned[[col]], levels = levels(train_data_cleaned[[col]]))
}

# Verify the structure of the cleaned test data
str(test_data_cleaned)


# Predict on test data
predictions_housing <- predict(final_model_housing, newdata = test_data)

# Actual prices
actual_prices <- test_data$price

# Metrics
mse_housing <- mean((predictions_housing - actual_prices)^2)
rsq_housing <- 1 - sum((predictions_housing - actual_prices)^2) / sum((actual_prices - mean(actual_prices))^2)

cat("Test Mean Squared Error:", mse_housing, "\n")
cat("Test R-squared:", rsq_housing, "\n")

plot(actual_prices, predictions_housing, main = "Actual vs Predicted Prices", 
     xlab = "Actual Price", ylab = "Predicted Price", pch = 16, col = "blue")
abline(0, 1, col = "red", lwd = 2)  # Line of perfect prediction


# Residuals vs Fitted and Q-Q plot

par(mfrow = c(2,2))
plot(model_initial, which = 1)
plot(model_initial, which = 2)

# Tests
bptest(model_initial)
vif(model_initial)
shapiro.test(resid(model_initial))
dwtest(model_initial)



# -----------------------------------------------
# Part III: Future work 
# -----------------------------------------------

# 1. Box-Cox Transformation
model_bc <- lm(price ~ ., data = train_data_cleaned)
bc_result <- boxcox(model_bc, lambda = seq(-2, 2, 0.1))
best_lambda <- bc_result$x[which.max(bc_result$y)]

if (abs(best_lambda) < 0.01) {
  train_data_cleaned$price_bc <- log(train_data_cleaned$price)
  test_data$price_bc <- log(test_data$price)
} else {
  train_data_cleaned$price_bc <- (train_data_cleaned$price^best_lambda - 1) / best_lambda
  test_data$price_bc <- (test_data$price^best_lambda - 1) / best_lambda
}

model_boxcox <- lm(price_bc ~ . -price, data = train_data_cleaned)
summary(model_boxcox)
# Histogram of Residuals
hist(resid(model_boxcox), breaks = 20, col = "lightblue", main = "Histogram of Residuals (Box-Cox)")

# 2. Log Transformation
train_data_cleaned$log_price <- log(train_data_cleaned$price)
test_data$log_price <- log(test_data$price)
model_log <- lm(log_price ~ . -price, data = train_data_cleaned)

# 3. Polynomial Regression (adding area^2)
train_data_cleaned$area2 <- train_data_cleaned$area^2
test_data$area2 <- test_data$area^2
poly_model <- lm(price ~ . + area2, data = train_data_cleaned)




bc_predictions <- predict(model_boxcox, newdata = test_data)
predicted_prices_bc <- if (abs(best_lambda) < 0.01) exp(bc_predictions) else (bc_predictions * best_lambda + 1)^(1 / best_lambda)

RMSE_bc <- sqrt(mean((predicted_prices_bc - actuals)^2))
RMSE_log <- sqrt(mean((exp(predict(model_log, newdata = test_data)) - actuals)^2))
RMSE_poly <- sqrt(mean((predict(poly_model, newdata = test_data) - actuals)^2))


cat("\nModel Comparisons:\n")
cat("Box-Cox Model RMSE:", RMSE_bc, "\n")
cat("Log Model RMSE:", RMSE_log, "\n")
cat("Polynomial Model RMSE:", RMSE_poly, "\n")






# Histogram of Residuals
hist(resid(model_boxcox), breaks = 20, col = "lightblue", main = "Histogram of Residuals (Box-Cox)")

# Breusch-Pagan Test
bptest(model_boxcox)

# Shapiro-Wilk Test
shapiro.test(resid(model_boxcox))

# Durbin-Watson Test
dwtest(model_boxcox)

# Influence Diagnostics
student_resid <- rstudent(model_boxcox)
leverage <- hatvalues(model_boxcox)

dffits_val <- dffits(model_boxcox)



import optuna
from optuna import Trial
import logging

from demo import LoraTrainingArguments, train_lora

logger = logging.getLogger(__name__)

# Optuna Objective function for optimization
def objective(trial: Trial):
    model_id = 'Qwen/Qwen1.5-1.8B'  # Replace with your actual model ID
    context_length = 512  # Assuming a context length of 512

    # Suggest hyperparameters using Optuna
    per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 1, 16)
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 16)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    lora_rank = trial.suggest_int("lora_rank", 2, 16)
    lora_alpha = trial.suggest_int("lora_alpha", 2, 16)
    lora_dropout = trial.suggest_float("lora_dropout", 0.1, 0.5)

    # Define the training arguments using the suggested values
    training_args = LoraTrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )

    # Train the model using the suggested hyperparameters
    try:
        eval_loss = train_lora(
            model_id=model_id,
            context_length=context_length,
            training_args=training_args
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return float("inf")  # Return a large loss in case of error

    # Return eval_loss to Optuna, which will use it for optimization
    return eval_loss


# Run Optuna optimization
def run_optuna():
    # Create Optuna study to minimize eval_loss
    study = optuna.create_study(direction="minimize")

    # Optimize using 20 trials (you can increase this for better search)
    study.optimize(objective, n_trials=20)

    # Output the best hyperparameters
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best evaluation loss: {study.best_value}")


if __name__ == "__main__":
    # Run the Optuna optimization
    run_optuna()

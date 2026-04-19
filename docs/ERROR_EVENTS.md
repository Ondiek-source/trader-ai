# ALL APP LEVEL ERROR EVENTS

## Config Error Events

- CONFIG_MISSING
- CONFIG_INVALID_VALUE
- CONFIG_TYPE_ERROR
- CONFIG_VALIDATION_ERROR
- CONFIG_PAIR_ERROR
- CONFIG_LENGTH_MISMATCH
- CONFIG_EXPIRY_ERROR
- CONFIG_PORT_ERROR
- CONFIG_RANGE_ERROR
- CONFIG_CEILING_ERROR
- CONFIG_REQUIRED_ERROR

## Config Informational Events

- CONFIG_OTC_NORMALIZE
- CONFIG_LIVE_MODE
- CONFIG_PRACTICE_MODE

## Pipeline Events

- PIPELINE_STARTUP_INITIATED - Pipeline boot sequence started
- PIPELINE_EXIT - Pipeline shutting down
- PIPELINE_SHUTDOWN_INITIATED - Graceful shutdown initiated
- SHUTDOWN_COMPLETE - All engines and tasks stopped
- DASHBOARD_UPDATED_SHUTDOWN - Dashboard status updated with stopped flag
- STAGE_COMPLETE - A boot stage completed successfully
- HISTORIAN_SYNC - Historical data sync completed for a symbol
- MODEL_PULLED_TO_LOCAL_DISK - Model artifact downloaded from cloud
- MODEL_LOADED - Model loaded into memory
- MODEL_INJECTED - Model injected into LiveEngine
- DASHBOARD_STARTED - HTTP dashboard server started
- TELEGRAM_BOT_STARTED - Telegram bot polling started
- TASK_GROUP_STARTED - All background tasks launched
- QUOTEX_STATUS_DISABLED - Quotex polling disabled
- ELAPSED_TIME_CALC_ERROR - Failed to calculate session elapsed time
- QUOTEX_STATUS_POLL_ERROR - Error polling Quotex status
- DAILY_REPORT_DISABLED - Daily reports disabled
- DAILY_REPORT_SCHEDULED - Next daily report time calculated
- DAILY_REPORT_SENT - Daily report sent successfully
- DAILY_REPORT_FAILED - Failed to send daily report
- ENGINE_CRASHED - LiveEngine crashed
- ENGINE_CRASHED_UNEXPECTED - LiveEngine crashed with unknown error
- RETRAIN_FIRST_BOOT - First boot checking for missing models
- RETRAIN_CYCLE_STARTED - Scheduled retrain cycle began
- RETRAIN_SKIPPED - Retrain skipped due to invalid expiry
- RETRAIN_NO_DATA - No bar data available for retrain
- RETRAIN_MODEL_EXISTS - Model already exists on first boot
- RETRAIN_TRAINING_START - Training started for a model
- RETRAIN_TRAINING_COMPLETE - Training completed for a model
- RETRAIN_MODEL_FAILED - Training failed for a model
- RETRAIN_NO_BEST_MODEL - No best model in registry after training
- RETRAIN_COMPLETE - Retrain cycle completed for a symbol
- RETRAIN_SYMBOL_FAILED - Retrain failed for a symbol
- ENGINE_RELOADED - Engine successfully reloaded new model
- ENGINE_RELOAD_FAILED - Engine failed to reload new model
- RETRAIN_CYCLE_COMPLETE - Full retrain cycle finished

## Dashboard Activity Events

- Quotex not connected - Quotex connection lost
- Quotex connected - Quotex connection restored
- Retrain cycle started - Retrain cycle began
- Retrain complete - Retrain finished for a symbol
- Engine crashed - Engine crashed
- Engine reloaded - Engine reloaded new model
- Engine reload failed - Engine reload failed
- Daily report sent - Daily report sent
- Daily report failed - Daily report failed


```
Trader AI
├─ .dockerignore
├─ create_test_structure.ps1
├─ deploy
│  ├─ dashboard-url.sh
│  ├─ deploy.sh
│  ├─ diagnose.sh
│  ├─ get-logs.sh
│  ├─ healthcheck.sh
│  ├─ killswitch.sh
│  ├─ provision.sh
│  ├─ teardown.sh
│  └─ watch-signals.sh
├─ docker-compose.yml
├─ Dockerfile
├─ docs
│  ├─ Backlog.md
│  ├─ Core
│  │  ├─ Config
│  │  │  ├─ Config.md
│  │  │  └─ Config_Test_Harness.md
│  │  └─ o.bash
│  ├─ instruction-manual.md
│  └─ Issues.bash
├─ src
│  ├─ core
│  │  ├─ config.py
│  │  ├─ dashboard.py
│  │  ├─ logger.py
│  │  ├─ log_storage.py
│  │  ├─ pipeline.py
│  │  ├─ storage.py
│  │  └─ __init__.py
│  ├─ data_engine
│  │  ├─ backfill.py
│  │  ├─ features.py
│  │  └─ __init__.py
│  ├─ main.py
│  ├─ ml_engine
│  │  ├─ model.py
│  │  ├─ model_manager.py
│  │  ├─ trainer.py
│  │  └─ __init__.py
│  ├─ trading
│  │  ├─ quotex_reader.py
│  │  ├─ quotex_stream.py
│  │  ├─ reporter.py
│  │  ├─ signals.py
│  │  ├─ twelveticks_stream.py
│  │  ├─ webhook.py
│  │  └─ __init__.py
│  └─ __init__.py
└─ tests
   ├─ conftest.py
   ├─ core
   │  ├─ config
   │  │  ├─ test_helpers.py
   │  │  ├─ test_load_config.py
   │  │  ├─ test_properties.py
   │  │  ├─ test_settings.py
   │  │  ├─ test_validation.py
   │  │  └─ __init__.py
   │  └─ __init__.py
   ├─ data_engine
   │  ├─ test_placeholder.py
   │  └─ __init__.py
   ├─ ml_engine
   │  ├─ test_placeholder.py
   │  └─ __init__.py
   ├─ trading
   │  ├─ test_placeholder.py
   │  └─ __init__.py
   └─ __init__.py

```
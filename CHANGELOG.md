# Changelog

All notable changes to this project will be documented in this file.

## [0.3.4] - 2024-05-01

### 🐛 Bug Fixes

- Fix serialization of thread.created_at

## [0.3.3] - 2024-05-01

### 🚜 Refactor

- Add thread desc & created_at in emit

## [0.3.2] - 2024-05-01

### 🐛 Bug Fixes

- Fix CreateError 'project is required'

## [0.3.1] - 2024-05-01

### 🐛 Bug Fixes

- Fixed datetime serialization

## [0.3.0] - 2024-05-01

### 🚀 Features

- Add description to thread schema

### 🚜 Refactor

- Use first chat as thread title

### ⚙️  Miscellaneous Tasks

- Increase length of thread name

## [0.2.7] - 2024-04-15

### ⚙️  Miscellaneous Tasks

- Add step to get release version

## [0.2.6] - 2024-04-15

### 🚜 Refactor

- Simplify task loading in Project class

## [0.2.5] - 2024-04-15

### ⚙️  Miscellaneous Tasks

- Update release workflow for new features

## [0.2.4] - 2024-04-15

### 🚜 Refactor

- Improve task saving efficiency

## [0.2.3] - 2024-04-14

### 🚜 Refactor

- Update get_version function to a separate module

## [0.2.2] - 2024-04-14

### 🚜 Refactor

- Update version retrieval in __init__.py

## [0.2.1] - 2024-04-14

### 🐛 Bug Fixes

- Update GitHub token in release workflow
- Handle  exception when loading tasks

## [0.2.0] - 2024-04-14

### 🚀 Features

- Add release workflow, template, and script
- Add release target for project deployment
- Add function to retrieve version from pyproject.toml

## [0.1.0] - 2024-04-14

### 🚀 Features

- Added makefile
- Add Authentication & Db management
- Add Alexis CLI component
- Add CORS support
- Add environment switcher to config.py
- Add Redis integration
- Add routes for project upload
- Add project and task prompts for alexis chatbot
- Refactor db to use context session
- Add middleware support
- Add langsmith tracing
- Added main chain
- *(cli)* Added shell command
- Add authentication
- Add optional parameter to include tasks in get_project
- *(components)* Add is_authenticated function for auth
- Add chat functionality to API router
- Add essential settings validator and set default env
- Add mysqlclient dependency
- Add Procfile with uvicorn configuration
- Add project routes and models
- Add endpoint to get authenticated user details
- Use Elements as API docs
- Add functions to get all projects and tasks
- Add new configuration options
- Add tools for project and task details
- Add token cost computation
- Add ipykernel dependency
- Update project description and setup instructions
- *(deps)* Add httpx_sse as dependency
- Add chat history tracking functionality
- Add socketio component and mount to app
- Add socket functionality for Alexis chat
- Added migrations using alembic
- Add release command to Procfile
- Reenabled langserve endpoints
- Add user name to chain metadata
- Add CORS and authentication settings for SocketIO
- Add Session Middleware
- Add MongoDB support
- Created alternative models based on mongodb
- Add sql to mongo migration script
- Add mistletoe for markdown parsing
- Add jupyter notebook as dev dependency
- Added storage feature for contexts
- Add `count_tokens` function to utils
- Add preprocessor for project's description
- Implement formating for project and task
- Add user-friendly thread titles
- Add projection + partial updates

### 🐛 Bug Fixes

- Prevent adding duplicate middlewares
- Update Python version to 3.10.13
- *(type)* Fixed type error
- Ensure project exists before thread creation
- Remove unused package 'alembic' from pyproject.toml
- Correct regex pattern group syntax in extract_tasks function
- Return correct variable in ProjectContext
- Project saving logic
- Typo + type hints
- Typo in Task class error message

### 🚜 Refactor

- Remove JWT model import
- Moved chain components to chat
- Improve user retrieval logic
- Update chain loading process in AlexisApp
- Update get_token and is_authenticated to be async
- Remove unnecessary import of is_authenticated
- Improve project id extraction performance
- Remove unnecessary tools from the list
- Update task formatting and extraction logic
- *(type)* Update input type for AlexChain
- Use sqlachemy scoped_session
- Improve BaseQuery functionality
- Update db import to include session in shell function
- Uid of models return str(id) instead
- Update socket connection handling and session management
- Update session to use custom scope function
- Refactor socketio.py for session management
- Refactor code to use MongoDB models
- Use the new contexts
- Generate unique thread titles
- Rename contexts
- Project existence check
- Simplify task existence logic
- Update preprocess_project function

### 🎨 Styling

- Ensure UUID columns are not nullable
- Update log settings for development and testing environments
- Update project.txt with markdown formatting
- Fix variable assignment syntax error
- Update type hint for 'history' field (#2)
- Update type hint for 'history' field
- Update logging configuration for verbosity
- Fix variable naming in for loop

### ⚙️  Miscellaneous Tasks

- Setup project
- Ignore schema and data files
- Fixed app startup
- Ignore database files
- Add pyjwt and sqlalchemy dependencies
- Added segments for different environments
- Improved database ssion availability
- Export OpenAI key to env
- Gather model in one place
- Remove unused code
- Update chains in config
- Refactor auth views
- Update user model
- *(editor)* Update default formatter
- Removed unused template files
- Remove chatbot prompt templates
- Update .gitignore for node files and script
- Separated concerns in database component
- Removed redundant function 'execute'
- Define all exports in alexis.models
- Deprecated BaseModel.get function
- *(dep)* Added mongoengine
- Remove migrations files
- Remove sql to mongo migration script
- Update dependencies and sentry integrations
- Removed old SQL models
- Updated references to old models
- Remove unused code related to sql
- Remove default limit in old redis context storage
- Add storage and contexts to shell context

<!-- generated by git-cliff -->

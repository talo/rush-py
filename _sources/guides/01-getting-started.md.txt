# Getting Started

## Access

First, if you don't have a Rush account, [go create one on the Rush platform's website](https://rush.cloud/login)!

### Rush Token

Once you've created your account and logged in, click the "Overview" tab in the left sidebar and create or obtain your API token on the [accounts page](https://rush.cloud/me). This is your "Rush Token", and will let you authenticate your access to Rush from your rush-py scripts.

### Rush Projects

You'll also need to create a project and obtain its ID. From the [projects page](https://rush.cloud/projects) (which is also the landing page after logging in), create a project using the "New Project" button. Enter a name for your project and click "Continue". You will now be taken to a page with a URL of the following structure: `https://rush.cloud/project/{PROJECT_ID}/workspace/{WORKSPACE_ID}`. This project's ID is the value at the part of the URL referenced by the `{PROJECT_ID}` place-holder above. Each run you do in Rush is associated with a project and will appear as part of that project when you view that project in the web interface, and runs done through rush-py are no different.

## Install rush-py

Install rush-py using `pip install rush-py`. Note that versions prior to 6.0 have a very different design that isn't covered by these docs, so make sure you're on at least that major version!

It's also possible to install using `uv`, `pdm`, `poetry`, or any other tool of your choice, since rush-py is [available on PyPI](https://pypi.org/project/rush-py).

## Authenticating and Setting a Project

Use environment variables to configure access:
- `RUSH_TOKEN`: Put your authentication token's value here.
- `RUSH_PROJECT`: Put your project's ID here.

On Linux/macOS, set these with:
```bash
export RUSH_TOKEN="your_token_here"
export RUSH_PROJECT="your_project_id"
```

On Windows PowerShell, set these with:
```powershell
$env:RUSH_TOKEN = "your_token_here"
$env:RUSH_PROJECT = "your_project_id"
```

There's one more environment variable, which would only need to be used if requested by a QDX employee:
- `RUSH_ENDPOINT`: Use this to choose which Rush server to use; if omitted, defaults to our production server at https://rush.cloud. To use a different server, you'll need to be given access to it explicitly.

Next, continue to [Doing Your First rush-py Run](./02-first-run.md).

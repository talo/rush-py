# Developing

## Creating a development environment
```
# with nix
nix develop

# or with poetry
poetry develop
```

## Adding dependencies - 
Please keep runtime dependencies to a minimum, and try not to use compiled/native dependencies to reduce 
multi-platform support burden.

```
poetry add -- dependency 
```

Add development dependencies to the dev group

## Updating Graphql queries/mutations

We auto-generate client functions from the Rush graphql schema.

Step 1. download the latest SDL from the playground #TODO: set up some form of schema-fetching endpoint

Step 2. replace `schema.graphql` in the root of this repo with the new version

Step 3. Copy the queries from rush-client into `combined.graphql` - we try to keep `tengu-client` in sync with
        rush-py to reduce feature divergence and unify the interaction approaches.
        I use `cat ../tengu/tengu-client/queries/* > combined.graphql` to combine everything into one file
        that ariadne-codegen understands

Step 4. Run `ariadne-codegen` in an environment with it installed to update the deps

Step 5. Update the `rush/provider.py` to address any incompatibilities / add helpers for new features


## Publishing

Step 1. Update all of the notebooks in nb/* to reflect changes

Step 2. Run pdoc to document the api `pdoc ./rush/doc.py -o ./nbs/api`

Step 3. Fix pathing incompatabilities with `quarto` - `mv ./nbs/api/rush/* ./nbs/api/ && cp ./nbs/api/doc.html ./nbs/api/index.html `

Step 4. Run the following `nbdev` commands for generating final documentation (if notebooks have changed)
```
nbdev_prepare
nbdev_docs
nbdev_readme
nbdev_preview # check that to docs look sane
```
Step 5. ensure relative links in readme gets fixed to point to their deployed version # TODO: automate this

Step 6. Increment package version in `pyproject.toml`

Step 7. `poetry build` & `poetry publish` (ensure you have set up your pypi token https://python-poetry.org/docs/repositories/)

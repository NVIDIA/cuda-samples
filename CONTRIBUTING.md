
# Contributing to the CUDA Samples

Thank you for your interest in contributing to the CUDA Samples!


## Getting Started

1. **Fork & Clone the Repository**:

   Fork the reporistory and clone the fork. For more information, check [GitHub's documentation on forking](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) and [cloning a repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository).

## Making Changes

1. **Create a New Branch**:

   ```bash
   git checkout -b your-feature-branch
   ```

2. **Make Changes**.

3. **Build and Test**:

   Ensure changes don't break existing functionality by building and running tests.

   For more details on building and testing, refer to the [Building and Testing](#building-and-testing) section below.

4. **Commit Changes**:

   ```bash
   git commit -m "Brief description of the change"
   ```

## Building and Testing

For information on building a running tests on the samples, please refer to the main [README](README.md)

## Creating a Pull Request

1. Push changes to your fork
2. Create a pull request targeting the `master` branch of the original CUDA Samples repository. Refer to [GitHub's documentation](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) for more information on creating a pull request.
3. Describe the purpose and context of the changes in the pull request description.

## Code Formatting (pre-commit hooks)

The CUDA Samples repository uses [pre-commit](https://pre-commit.com/) to execute all code linters and formatters. These
tools ensure a consistent coding style throughout the project. Using pre-commit ensures that linter
versions and options are aligned for all developers. Additionally, there is a CI check in place to
enforce that committed code follows our standards.

The linters used by the CUDA Samples are listed in `.pre-commit-config.yaml`.
For example, C++ and CUDA code is formatted with [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html).

To use `pre-commit`, install via `conda` or `pip`:

```bash
conda config --add channels conda-forge
conda install pre-commit
```

```bash
pip install pre-commit
```

Then run pre-commit hooks before committing code:

```bash
pre-commit run
```

By default, pre-commit runs on staged files (only changes and additions that will be committed).
To run pre-commit checks on all files, execute:

```bash
pre-commit run --all-files
```

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit. This can be done by running:

```bash
pre-commit install
```

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with `git commit --no-verify` or with the short version `git commit -n`, althoguh please note
that this may result in pull requests being rejected if subsequent checks fail.

## Review Process

Once submitted, maintainers will be automatically assigned to review the pull request. They might suggest changes or improvements. Constructive feedback is a part of the collaborative process, aimed at ensuring the highest quality code.

For constructive feedback and effective communication during reviews, we recommend following [Conventional Comments](https://conventionalcomments.org/).

Further recommended reading for successful PR reviews:

- [How to Do Code Reviews Like a Human (Part One)](https://mtlynch.io/human-code-reviews-1/)
- [How to Do Code Reviews Like a Human (Part Two)](https://mtlynch.io/human-code-reviews-2/)

## Thank You

Your contributions enhance the CUDA Samples for the entire community. We appreciate your effort and collaboration!

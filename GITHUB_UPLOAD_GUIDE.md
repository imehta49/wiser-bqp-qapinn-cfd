# Uploading This Project to GitHub

Recommended repository name:

```text
wiser-bqp-qapinn-cfd
```

Recommended description:

```text
Explainable classical and quantum-assisted PINNs for Burgers, heat, and reduced projectile CFD benchmarks — WISER/BQP 2026.
```

## Method A — Terminal and Git

### 1. Create an empty repository on GitHub

On GitHub, select **New repository**, enter the name above, and choose public or private visibility. Because this project folder already includes a README, `.gitignore`, and license, do not initialize the remote repository with additional files.

### 2. Extract the downloaded repository ZIP

Extract `wiser-bqp-qapinn-cfd-github-ready.zip` on your computer, then open a terminal inside the extracted folder.

### 3. Configure Git once

```bash
git config --global user.name "YOUR FULL NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

### 4. Initialize and inspect

```bash
git init
git status
```

### 5. Commit the initial repository

```bash
git add .
git commit -m "Initial classical Burgers PINN baseline"
git branch -M main
```

### 6. Connect the GitHub repository

Replace `YOUR_USERNAME`:

```bash
git remote add origin https://github.com/YOUR_USERNAME/wiser-bqp-qapinn-cfd.git
git remote -v
```

### 7. Push

```bash
git push -u origin main
```

Use GitHub authentication, GitHub CLI, or an SSH key. Never paste an access token into a notebook, source file, or committed configuration file.

## Method B — GitHub Desktop

1. Extract the repository ZIP.
2. Open GitHub Desktop.
3. Choose **File → Add local repository**.
4. Select the extracted `wiser-bqp-qapinn-cfd` folder.
5. If requested, choose **Create a repository here**.
6. Enter the first commit message.
7. Click **Commit to main**.
8. Click **Publish repository**.
9. Choose the repository visibility and publish.

## Normal update workflow

After changing code or adding results:

```bash
git status
git add .
git commit -m "Describe the experiment or correction"
git push
```

Useful commit examples:

```text
Add seeds 1 and 2 for smooth Burgers baseline
Add parameter-matched classical PINN
Implement four-qubit QAPINN
Add Fourier-spectrum analysis
Add reduced projectile CFD dataset
```

## Branch workflow for team members

```bash
git checkout -b feature/qapinn-4qubit
# edit and test files
git add .
git commit -m "Implement four-qubit QAPINN"
git push -u origin feature/qapinn-4qubit
```

Open a pull request on GitHub and merge after review.

## Large result files

Do not commit large model checkpoints, raw arrays, CFD meshes, or generated ZIP archives to ordinary Git history. The included `.gitignore` excludes these files. Use a GitHub Release, Git LFS, or an archival data repository when large artefacts must be shared.

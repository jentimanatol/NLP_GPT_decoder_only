@echo off
git --version

:: Create a new orphan branch
git checkout --orphan new-branch

:: Add all files to the new branch
git add -A

:: Commit the changes
git commit -m "Initial commit (clean history)"

:: Delete the old main branch
git branch -D main

:: Rename the new branch to main
git branch -m main

:: Force push the new branch to the remote repository
git push -f origin main

pause
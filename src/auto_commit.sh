#!/bin/sh

cd /home/mciglenecki/projects/nerf-research
git add -u
git qcommit
git push

cd /home/mciglenecki/projects/nerf-research/nerfstudio
git add -u
git qcommit
git push
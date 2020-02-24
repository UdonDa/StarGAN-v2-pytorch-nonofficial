rsync -davzru --include-from <(git ls-files) --exclude .git --exclude-from <(git ls-files -o --directory) . im00:/export/space0/horita-d/conf/eccv2020/centermask/

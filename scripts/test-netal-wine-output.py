#!/usr/bin/env python3
"""A child retaining both output handles must not delay the finished command's result."""
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='Netal output ') as temporary:
    directory = Path(temporary)
    child = directory/'hold.py'
    child.write_text('''from pathlib import Path
import sys,time
release=Path(sys.argv[1]); done=Path(sys.argv[2]); deadline=time.monotonic()+20
while not release.exists() and time.monotonic()<deadline:time.sleep(.02)
done.write_text('done')
''')
    parent = directory/'parent.py'
    parent.write_text('''import subprocess,sys
subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2],sys.argv[3]])
print('mac_pid=12345',flush=True)
print('controlled diagnostic',file=sys.stderr,flush=True)
sys.exit(int(sys.argv[4]))
''')
    for status in (0,7):
        release=directory/f'release-{status}'
        done=directory/f'done-{status}'
        command=['zsh','-c','source "$1"; shift; netal_wine_capture "$@"','test',
                 str(root/'scripts/netal-wine-output.sh'),sys.executable,str(parent),str(child),str(release),str(done),str(status)]
        process=subprocess.Popen(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            stdout,stderr=process.communicate(timeout=8)
            assert not done.exists(), 'Child exited before result delivery; the test did not establish independence'
            assert process.returncode==status and stdout=='mac_pid=12345\n', (process.returncode,stdout,stderr)
            assert stderr==('controlled diagnostic\n' if status else ''), stderr
        finally:
            release.touch()
            if process.poll() is None: process.communicate(timeout=8)
            # The fixture owns this completion marker; no process-name search or broad kill.
            import time
            deadline=time.monotonic()+5
            while not done.exists() and time.monotonic()<deadline:time.sleep(.02)
            assert done.exists(), 'Owned output-holding fixture did not finish'
print('Wine output capture passes success/failure with inherited output handles')

#!/usr/bin/env python3
"""Preview or explicitly apply the checked-in ruleset; never change visibility."""
import argparse,json,re,subprocess,sys
from pathlib import Path

def api(path,*args):
    result=subprocess.run(['gh','api',path,*args],capture_output=True,text=True)
    if result.returncode:
        if 'Upgrade to GitHub Pro' in result.stderr or 'HTTP 403' in result.stderr:
            raise RuntimeError('GitHub refused ruleset access. The current private-repository plan may not support enforcement. Visibility was not changed.')
        raise RuntimeError('GitHub API request failed; inspect authentication and repository permissions. No credentials are printed.')
    return json.loads(result.stdout) if result.stdout.strip() else None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,help='owner/name')
    parser.add_argument('--apply',action='store_true',help='Create/update this one named ruleset after verifying checks')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repo):parser.error('Use owner/name')
    path=Path(__file__).resolve().parents[1]/'.github/rulesets/main.json';spec=json.loads(path.read_text())
    if not args.apply:
        print(json.dumps({'mode':'preview-only','repo':args.repo,'ruleset':spec,'visibility_change':False},indent=2));return 0
    repo=api('repos/'+args.repo);before=repo['visibility']
    existing=api('repos/'+args.repo+'/rulesets')
    matches=[r for r in existing if r['name']==spec['name']]
    if len(matches)>1:raise RuntimeError('Multiple rulesets have the target name; reconcile manually.')
    checks=api('repos/'+args.repo+'/commits/'+repo['default_branch']+'/check-runs')['check_runs']
    required=next(r for r in spec['rules'] if r['type']=='required_status_checks')['parameters']['required_status_checks']
    for check in required:
        entries=[r for r in checks if r['name']==check['context'] and r['app']['id']==check['integration_id']]
        if not entries or max(entries,key=lambda r:r['id'])['conclusion']!='success':
            raise RuntimeError('Required check is missing or not successful: '+check['context'])
    target='repos/'+args.repo+'/rulesets'+('/'+str(matches[0]['id']) if matches else '')
    result=api(target,'--method','PUT' if matches else 'POST','--input',str(path))
    actual=api('repos/'+args.repo+'/rulesets/'+str(result['id']))
    if actual['enforcement']!='active':raise RuntimeError('Ruleset saved but enforcement is not active.')
    if api('repos/'+args.repo)['visibility']!=before:raise RuntimeError('Repository visibility changed concurrently; inspect it immediately.')
    print(json.dumps({'mode':'applied','id':actual['id'],'name':actual['name'],'enforcement':actual['enforcement'],'visibility':before},indent=2))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except RuntimeError as error:print(str(error),file=sys.stderr);raise SystemExit(1)

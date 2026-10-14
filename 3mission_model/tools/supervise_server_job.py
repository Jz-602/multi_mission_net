"""单任务后台等待和固定 Codex 会话事件投递；不自行验收或重跑实验。"""
import argparse
import datetime
import fcntl
import glob
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

import websocket


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
STATE_ROOT = PROJECT / 'log/server_supervision'
THREAD_ID = '01a0f2e7-f1cc-73b2-bbed-552da600efd7'
SOCKET_PATH = Path.home() / '.codex/app-server-control/app-server-control.sock'


def Now():
    return datetime.datetime.now().astimezone().isoformat()


def Read(path):
    return json.loads(path.read_text())


def ResolveProgressGlob(pattern):
    """log/...以3mission_model为基准，其余相对路径沿用仓库根目录。"""
    path = Path(pattern)
    base = PROJECT if path.parts and path.parts[0] == 'log' else ROOT
    return str(base / path)


def Write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def Identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        return {'pid': pid, 'start_ticks': (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19],
                'argv': [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]}
    except (FileNotFoundError, ProcessLookupError):
        return None


class CodexClient:
    def __init__(self):
        connection = socket.socket(socket.AF_UNIX)
        connection.settimeout(15)
        connection.connect(str(SOCKET_PATH))
        self.connection = websocket.create_connection('ws://localhost/', socket=connection, timeout=15)
        self.request_id = 0
        self.Call('initialize', {'clientInfo': {'name': 'multi_mission_supervisor', 'version': '1.0'},
                                 'capabilities': {'experimentalApi': True}})
        self.connection.send(json.dumps({'method': 'initialized'}))

    def Call(self, method, params):
        self.request_id += 1
        self.connection.send(json.dumps({'id': self.request_id, 'method': method, 'params': params}))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            response = json.loads(self.connection.recv())
            if response.get('id') == self.request_id:
                if 'error' in response:
                    raise RuntimeError(json.dumps(response['error'], ensure_ascii=False))
                return response['result']
        raise TimeoutError(method)

    def Goal(self):
        return self.Call('thread/goal/get', {'threadId': THREAD_ID})['goal']

    def Close(self):
        self.connection.close()


def Enabled(client):
    control = STATE_ROOT / 'control.json'
    if control.exists() and Read(control).get('mode') != 'active':
        return False
    goal = client.Goal()
    return goal is not None and goal['status'] == 'active'


def Record(job_path, **fields):
    record = Read(job_path)
    record.update(updated_at=Now(), **fields)
    Write(job_path, record)
    return record


def Progress(message):
    with (PROJECT / 'SERVER_PROGRESS.md').open('a') as stream:
        stream.write('\n- 后台监督 ' + Now() + '：' + message + '\n')


def RegisteredWaitingTurn(job_path, client):
    registration_path = STATE_ROOT / 'event_wait.json'
    if not registration_path.exists():
        return None
    try:
        registration = Read(registration_path)
    except (OSError, ValueError):
        return None
    if (registration.get('job_path') != str(job_path.resolve())
            or registration.get('thread_id') != THREAD_ID
            or not registration.get('expected_turn_id')):
        return None
    turns = client.Call('thread/turns/list', {'threadId': THREAD_ID, 'limit': 1})['data']
    if (not turns or turns[0]['id'] != registration['expected_turn_id']
            or turns[0]['status'] != 'inProgress'):
        return None
    return registration['expected_turn_id']


def Notify(job_path, kind, client, b_wait=True):
    """一次事件只入队一次；投递结果不确定时留证阻塞，不盲目重复发送。"""
    job = Read(job_path)
    event_path = job_path.parent / ('event_' + kind + '.json')
    if not event_path.exists():
        event = {'event_id': str(uuid.uuid4()), 'kind': kind, 'created_at': Now(),
                 'job_path': str(job_path), 'thread_id': THREAD_ID, 'status': 'prepared'}
        Write(event_path, event)
    event = Read(event_path)
    if event['status'] in ('started', 'steered', 'acknowledged', 'dispatched_by_native_queue'):
        return
    if event['status'] == 'sending':
        raise RuntimeError('上次投递响应不确定；保留现场，人工核对投递记录后处理，禁止重复发送')
    if not Enabled(client):
        if event['status'] == 'queued':
            client.Call('thread/queue/delete', {'threadId': THREAD_ID, 'queuedSubmissionId': event['queue_id']})
        event.update(status='held', held_at=Now())
        Write(event_path, event)
        Record(job_path, notification_status='held_by_pause_or_goal_status')
        return
    if event['status'] == 'held':
        Record(job_path, notification_status='held_requires_explicit_dispatch')
        return
    if event['status'] == 'prepared':
        message = ('[multi_mission 后台事件] ' + kind + '；事件文件 ' + str(event_path) +
                   '。固定原会话自动续接。先读取事件/job及SERVER_PROGRESS，核对实际进程；'
                   '退出不等于成功。依约验收产物或定位错误，不盲重跑。若为自测，按job.next_step执行并留证；'
                   '处理完使用supervise_server_job.py ack --event <事件路径> --evidence <验收JSON路径>。'
                   '人工暂停/取消时不得推进；额度不足不重置。')
        if kind.startswith('milestone_'):
            message += '这是运行中里程碑：检查首轮ETA或对应完整验证/图片，只ack该事件，保持当前训练，不启动下一阶段。'
        waiting_turn_id = RegisteredWaitingTurn(job_path, client)
        event.update(status='sending', send_attempt_at=Now(),
                     delivery='steer' if waiting_turn_id else 'queue')
        if waiting_turn_id:
            event['expected_turn_id'] = waiting_turn_id
        Write(event_path, event)
        if waiting_turn_id:
            result = client.Call('turn/steer', {'threadId': THREAD_ID,
                                 'expectedTurnId': waiting_turn_id,
                                 'clientUserMessageId': event['event_id'],
                                 'input': [{'type': 'text', 'text': message, 'text_elements': []}]})
            event.update(status='steered', turn_id=waiting_turn_id,
                         steer_result=result, steered_at=Now())
            Write(event_path, event)
            Record(job_path, notification_status='steered', event_path=str(event_path))
            Progress(f"{job['name']} {kind} 已投递显式等待turn；事件 {event_path}，尚未验收。")
            return
        result = client.Call('thread/queue/add', {'threadId': THREAD_ID,
                            'clientUserMessageId': event['event_id'],
                            'input': [{'type': 'text', 'text': message, 'text_elements': []}]})
        event.update(status='queued', queue_id=result['queuedSubmission']['id'], queued_at=Now())
        Write(event_path, event)
        Record(job_path, notification_status='queued', event_path=str(event_path))
        Progress(f"{job['name']} {kind} 已投递固定会话；事件 {event_path}，尚未验收。")

    ## 当前会话忙时只由普通程序等待；空闲后按确切队列 ID 唤醒。
    while event['status'] == 'queued':
        if (event_path.parent / (event_path.stem + '.ack.json')).exists():
            return
        if not Enabled(client):
            client.Call('thread/queue/delete', {'threadId': THREAD_ID, 'queuedSubmissionId': event['queue_id']})
            event.update(status='held', held_at=Now())
            Write(event_path, event)
            Record(job_path, notification_status='held_by_pause_or_goal_status')
            return
        thread = client.Call('thread/read', {'threadId': THREAD_ID, 'includeTurns': False})['thread']
        queue = client.Call('thread/queue/list', {'threadId': THREAD_ID})['data']
        if not any(item['id'] == event['queue_id'] for item in queue):
            event.update(status='dispatched_by_native_queue', dispatched_at=Now())
            Write(event_path, event)
            Record(job_path, notification_status=event['status'])
            return
        if thread['status']['type'] in ('idle', 'notLoaded'):
            result = client.Call('thread/queue/start', {'threadId': THREAD_ID,
                                                       'queuedSubmissionId': event['queue_id']})
            event.update(status='started', turn_id=result['turn']['id'], started_at=Now())
            Write(event_path, event)
            Record(job_path, notification_status='started')
            return
        if not b_wait:
            return
        time.sleep(job['dispatch_interval'])


def PollNotify(job_path, kind, client):
    """Network failure retains the local job; uncertain sends never replay."""
    event_path = job_path.parent / ('event_' + kind + '.json')
    try:
        if client is None:
            client = CodexClient()
        Notify(job_path, kind, client, b_wait=False)
    except Exception as error:
        if client is not None:
            try:
                client.Close()
            except Exception:
                pass
        client = None
        b_uncertain = event_path.exists() and Read(event_path)['status'] == 'sending'
        Record(job_path, notification_status='uncertain_send_requires_review' if b_uncertain else 'network_retry_pending',
               notification_error=repr(error))
        return client, b_uncertain
    event = Read(event_path)
    b_done = event['status'] in ('started', 'steered', 'acknowledged', 'dispatched_by_native_queue', 'held') or event_path.with_name(event_path.stem + '.ack.json').exists()
    return client, b_done


def PollMilestones(job_path, job, client):
    started = datetime.datetime.fromisoformat(job['started_at']).timestamp()
    for name, filename in job.get('milestones', {}).items():
        path = Path(filename)
        if path.is_file() and path.stat().st_mtime >= started:
            client, _ = PollNotify(job_path, 'milestone_' + name, client)
    return client


def Work(job_path):
    with (STATE_ROOT / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = CodexClient()
        job = Read(job_path)
        if job['status'] != 'created':
            raise RuntimeError('已有执行记录，拒绝重复启动')
        if not Enabled(client):
            Record(job_path, status='held_before_start')
            return
        env = dict(os.environ, **job['env'])
        with (job_path.parent / 'stdout.log').open('a') as stream:
            started_at = Now()
            child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            job = Record(job_path, status='running', started_at=started_at, supervisor=Identity(os.getpid()),
                         child=Identity(child.pid))
            Progress(f"启动 {job['name']}，job={job_path}，supervisor={os.getpid()}，child={child.pid}；{job['next_step']}")
            last_progress = time.time()
            previous_signature = None
            stale_checks = 0
            b_alert_pending = False
            last_milestone_check = time.monotonic()
            while True:
                try:
                    returncode = child.wait(timeout=job['health_interval'])
                    break
                except subprocess.TimeoutExpired:
                    paths = [job_path.parent / 'stdout.log']
                    for pattern in job['progress_globs']:
                        paths.extend(Path(filename) for filename in glob.glob(ResolveProgressGlob(pattern), recursive=True))
                    signature = sorted((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                                       for path in paths if path.is_file())
                    if signature != previous_signature:
                        last_progress, stale_checks = time.time(), 0
                    else:
                        stale_checks += 1
                    previous_signature = signature
                    free_bytes = shutil.disk_usage(ROOT).free
                    b_stalled = time.time() - last_progress > job['stall_seconds'] and stale_checks >= 2
                    Record(job_path, health_at=Now(), free_bytes=free_bytes,
                           seconds_without_progress=time.time() - last_progress)
                    try:
                        if client is None:
                            client = CodexClient()
                        goal = client.Goal()
                        Record(job_path, goal_status=goal['status'] if goal else None)
                    except Exception as error:
                        if client is not None:
                            try:
                                client.Close()
                            except Exception:
                                pass
                        client = None
                        Record(job_path, notification_status='network_retry_pending', notification_error=repr(error))
                    if not b_alert_pending and (b_stalled or free_bytes < 20 * 1024**3):
                        Record(job_path, alert='confirmed_no_progress' if b_stalled else 'low_disk')
                        b_alert_pending = True
                    if b_alert_pending:
                        client, _ = PollNotify(job_path, 'health_alert', client)
                    if time.monotonic() - last_milestone_check >= 300:
                        client = PollMilestones(job_path, job, client)
                        last_milestone_check = time.monotonic()
            Record(job_path, status='exited_pending_acceptance', returncode=returncode, ended_at=Now())
            b_done = False
            while not b_done:
                client, b_done = PollNotify(job_path, 'success_exit' if returncode == 0 else 'failed_exit', client)
                if not b_done:
                    time.sleep(job['dispatch_interval'])
        if client is not None:
            client.Close()


def Dispatch(job_path):
    """Resume only an already exited job's event; never start its command."""
    with (STATE_ROOT / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = Read(job_path)
        if job.get('status') != 'exited_pending_acceptance' or job.get('returncode') is None:
            raise RuntimeError('需要已落盘的真实退出码；禁止为运行/未启动任务投递完成事件')
        child = job.get('child')
        if child and Identity(child['pid']) == child:
            raise RuntimeError('子进程仍活动；禁止投递完成事件')
        kind = 'success_exit' if job['returncode'] == 0 else 'failed_exit'
        event_path = job_path.parent / ('event_' + kind + '.json')
        if event_path.exists() and Read(event_path)['status'] == 'sending':
            raise RuntimeError('发送结果不确定，需人工核对队列；禁止盲入队')
        client = CodexClient()
        if not Enabled(client):
            client.Close()
            raise RuntimeError('control和goal都必须active才允许显式dispatch')
        if event_path.exists() and Read(event_path)['status'] == 'held':
            event = Read(event_path)
            event.update(status='prepared', explicit_dispatch_at=Now())
            Write(event_path, event)
        b_done = False
        while not b_done:
            client, b_done = PollNotify(job_path, kind, client)
            if not b_done:
                time.sleep(job['dispatch_interval'])
        if client is not None:
            client.Close()


def Launch(args):
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with (STATE_ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current_path = STATE_ROOT / 'current.json'
        if current_path.exists():
            previous = Path(Read(current_path)['job_path'])
            old = Read(previous)
            if not (previous.parent / 'accepted.json').exists():
                raise RuntimeError(f'前一任务未验收/处理，禁止新启动：{previous}')
            for key in ('child', 'supervisor'):
                identity = old.get(key)
                if identity and Identity(identity['pid']) == identity:
                    raise RuntimeError(f'前一{key}仍在运行，禁止并发启动')
        client = CodexClient()
        if not Enabled(client):
            raise RuntimeError('人工控制或goal非active，禁止启动')
        client.Close()
        job_dir = STATE_ROOT / (datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + args.name)
        job_dir.mkdir()
        job_path = job_dir / 'job.json'
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            raise ValueError('缺少子进程命令')
        env = dict(pair.split('=', 1) for pair in args.env)
        milestones = {}
        for specification in args.milestone:
            name, filename = specification.split('=', 1)
            path = (PROJECT / filename).resolve()
            if (not name or not all(character.isalnum() or character in '_-' for character in name)
                    or Path(filename).is_absolute() or not path.is_relative_to(PROJECT)):
                raise ValueError('milestone需名称=相对3mission_model内的文件路径')
            if name in milestones:
                raise ValueError('milestone名称不可重复')
            milestones[name] = str(path)
        Write(job_path, dict(name=args.name, command=command, env=env, thread_id=THREAD_ID,
                            created_at=Now(), status='created', next_step=args.next_step,
                            health_interval=args.health_interval, dispatch_interval=args.dispatch_interval,
                            stall_seconds=args.stall_seconds,
                            progress_globs=[ResolveProgressGlob(pattern) for pattern in args.progress_glob],
                            milestones=milestones))
        Write(current_path, {'job_path': str(job_path)})
        with (job_dir / 'supervisor.log').open('a') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'work',
                                        '--job', str(job_path)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        Write(job_dir / 'launch.json', {'supervisor': Identity(process.pid), 'launched_at': Now()})
        print(json.dumps({'job_path': str(job_path), 'supervisor_pid': process.pid}))


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    launch = commands.add_parser('launch')
    launch.add_argument('--name', required=True)
    launch.add_argument('--next-step', required=True)
    launch.add_argument('--env', action='append', default=[])
    launch.add_argument('--progress-glob', action='append', default=[],
                        help='log/...相对3mission_model，其余相对仓库根目录；也接受绝对路径')
    launch.add_argument('--milestone', action='append', default=[], help='名称=相对3mission_model内的文件路径；可重复')
    launch.add_argument('--health-interval', type=int, default=300)
    launch.add_argument('--dispatch-interval', type=int, default=30)
    launch.add_argument('--stall-seconds', type=int, default=1800)
    launch.add_argument('command', nargs=argparse.REMAINDER)
    work = commands.add_parser('work')
    work.add_argument('--job', type=Path, required=True)
    dispatch = commands.add_parser('dispatch')
    dispatch.add_argument('--job', type=Path, required=True)
    ack = commands.add_parser('ack')
    ack.add_argument('--event', type=Path, required=True)
    ack.add_argument('--evidence', type=Path, required=True)
    control = commands.add_parser('control')
    control.add_argument('mode', choices=('active', 'paused', 'cancelled'))
    args = parser.parse_args()
    if args.action == 'launch':
        Launch(args)
    elif args.action == 'work':
        try:
            Work(args.job.resolve())
        except BaseException as error:
            Record(args.job, supervisor_status='blocked', supervisor_error=repr(error))
            Progress(f'监督程序阻塞 {args.job}：{error!r}；保留子进程/现场，不自行重跑或重置额度。')
            raise
    elif args.action == 'ack':
        event = Read(args.event)
        evidence = Read(args.evidence)
        if not evidence.get('b_handled') or evidence.get('event_id') != event['event_id']:
            raise ValueError('需要匹配事件ID且b_handled=true的真实验收/错误处理证据')
        record = dict(event_id=event['event_id'], evidence=str(args.evidence.resolve()), handled_at=Now())
        Write(args.event.with_name(args.event.stem + '.ack.json'), record)
        if event['kind'] in ('success_exit', 'failed_exit'):
            Write(args.event.parent / 'accepted.json', record)
        print(json.dumps(record, ensure_ascii=False))
    elif args.action == 'dispatch':
        Dispatch(args.job.resolve())
    elif args.action == 'control':
        Write(STATE_ROOT / 'control.json', {'mode': args.mode, 'updated_at': Now()})


if __name__ == '__main__':
    Main()

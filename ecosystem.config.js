module.exports = {
  apps: [{
    name: 'pregnancy-chatbot',
    script: 'chatbot_v0.1.py',
    interpreter: '/mnt/hdd_sda/.conda/envs/preg/bin/python',
    cwd: '/mnt/hdd_sda/projects/STEAM',
    instances: 1,
    autorestart: true,
    watch: false,
    // 대용량 LLM 로딩 시 RSS가 일시적으로 커질 수 있으므로 임계치 상향
    max_memory_restart: '32G',
    env: {
      NODE_ENV: 'production',
      CUDA_VISIBLE_DEVICES: '0,1',
      PYTHONPATH: '/mnt/hdd_sda/projects/STEAM',
      CONDA_DEFAULT_ENV: 'preg'
    },
    error_file: './logs/pm2-error.log',
    out_file: './logs/pm2-out.log',
    log_file: './logs/pm2-combined.log',
    time: true,
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    kill_timeout: 10000,
    restart_delay: 5000,
    max_restarts: 10,
    min_uptime: '10s'
  }]
};

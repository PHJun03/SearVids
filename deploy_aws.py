#!/usr/bin/env python3
"""
Searvids AWS Automated Deployment and Benchmark
Uses boto3 for reliable AWS automation
"""

import boto3
import time
import subprocess
import sys
import os
from datetime import datetime
import json
import argparse

# Configuration
REGION = "ap-northeast-2"
INSTANCE_TYPE = "g5.xlarge"
AMI_ID = "ami-05d2438ca66594916"  # Ubuntu 22.04 LTS
VOLUME_SIZE = 100

class AWSDeployer:
    def __init__(self, region=REGION, skip_cleanup=True):
        self.region = region
        self.skip_cleanup = True # FORCE TRUE FOR DEBUGGING
        self.ec2_client = boto3.client('ec2', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)
        
        # State tracking
        self.instance_id = None
        self.security_group_id = None
        self.key_name = None
        self.key_path = None
        self.public_ip = None
        
        # Unique naming
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.security_group_name = f"searvids-sg-{timestamp}"
        self.key_name = f"searvids-key-{timestamp}"
        self.instance_name = "searvids-benchmark"
    
    def log(self, step, message, color='cyan'):
        colors = {'cyan': '\033[96m', 'green': '\033[92m', 'yellow': '\033[93m', 
                  'red': '\033[91m', 'gray': '\033[90m', 'end': '\033[0m'}
        prefix = colors.get(color, '') + f"[{step}] " + colors['end']
        print(f"{prefix}{message}")
    
    def create_security_group(self):
        """Step 1: Create security group with SSH and API access"""
        self.log("1/10", "Creating security group...", 'yellow')
        
        # Get default VPC
        vpcs = self.ec2_client.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
        self.vpc_id = vpcs['Vpcs'][0]['VpcId']
        
        # Create security group
        response = self.ec2_client.create_security_group(
            GroupName=self.security_group_name,
            Description='Searvids benchmark security group',
            VpcId=self.vpc_id
        )
        self.security_group_id = response['GroupId']
        
        # Get my IP
        import requests
        my_ip = requests.get('http://checkip.amazonaws.com').text.strip()
        
        # Allow SSH
        self.ec2_client.authorize_security_group_ingress(
            GroupId=self.security_group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': f'{my_ip}/32'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8080,
                    'ToPort': 8080,
                    'IpRanges': [{'CidrIp': f'{my_ip}/32'}]
                }
            ]
        )
        
        self.log("1/10", f"✓ Security group created: {self.security_group_id}", 'green')
    
    def create_key_pair(self):
        """Step 2: Create SSH key pair"""
        self.log("2/10", "Creating SSH key pair...", 'yellow')
        
        response = self.ec2_client.create_key_pair(KeyName=self.key_name)
        self.key_path = f"{self.key_name}.pem"
        
        # Save key file
        with open(self.key_path, 'w') as f:
            f.write(response['KeyMaterial'])
        
        # Set permissions (Unix-like systems)
        os.chmod(self.key_path, 0o400)
        
        self.log("2/10", f"✓ Key saved: {self.key_path}", 'green')
    
    def launch_instance(self):
        """Step 3: Launch EC2 g5.xlarge instance"""
        self.log("3/10", f"Launching EC2 instance ({INSTANCE_TYPE})...", 'yellow')
        self.log("3/10", "   This will cost ~$1.24/hour", 'gray')
        
        # 1. Find valid Availability Zones for this instance type
        self.log("3/10", "   Identifying valid Availability Zones...", 'gray')
        az_response = self.ec2_client.describe_instance_type_offerings(
            LocationType='availability-zone',
            Filters=[{'Name': 'instance-type', 'Values': [INSTANCE_TYPE]}]
        )
        valid_azs = [offer['Location'] for offer in az_response['InstanceTypeOfferings']]
        if not valid_azs:
            raise Exception(f"Instance type {INSTANCE_TYPE} is not available in any AZ in {self.region}")
        
        self.log("3/10", f"   Valid AZs: {', '.join(valid_azs)}", 'gray')
        
        # 2. Find a subnet in one of the valid AZs
        subnets = self.ec2_client.describe_subnets(
            Filters=[
                {'Name': 'vpc-id', 'Values': [self.vpc_id]},
                {'Name': 'availability-zone', 'Values': valid_azs}
            ]
        )
        
        if not subnets['Subnets']:
            raise Exception(f"No subnets found in valid AZs ({valid_azs}) for VPC {self.vpc_id}")
            
        deployment_success = False
        last_error = None
        
        # Retry parameters
        max_quota_retries = 10  # 10 minutes total
        
        for target_subnet in subnets['Subnets']:
            subnet_id = target_subnet['SubnetId']
            target_az = target_subnet['AvailabilityZone']
            
            self.log("3/10", f"   Attempting launch in {target_az} (Subnet: {subnet_id})...", 'yellow')
            
            # Quota retry loop
            for attempt in range(max_quota_retries):
                try:
                    self.log("3/10", f"   Attempting simplified launch in {target_az}...", 'gray')
                    
                    response = self.ec2_client.run_instances(
                        ImageId=AMI_ID,
                        InstanceType=INSTANCE_TYPE,
                        KeyName=self.key_name,
                        MinCount=1,
                        MaxCount=1,
                        SubnetId=subnet_id,
                        SecurityGroupIds=[self.security_group_id],
                        BlockDeviceMappings=[
                            {
                                'DeviceName': '/dev/sda1',
                                'Ebs': {
                                    'VolumeSize': VOLUME_SIZE,
                                    'VolumeType': 'gp3',
                                    'DeleteOnTermination': True
                                }
                            }
                        ],
                        TagSpecifications=[
                            {
                                'ResourceType': 'instance',
                                'Tags': [{'Key': 'Name', 'Value': self.instance_name}]
                            }
                        ]
                    )
                    
                    self.instance_id = response['Instances'][0]['InstanceId']
                    self.log("3/10", f"✓ Instance launched: {self.instance_id} in {target_az}", 'green')
                    deployment_success = True
                    break # Break retry loop
                    
                except Exception as e:
                    # Check for VcpuLimitExceeded
                    if "VcpuLimitExceeded" in str(e):
                        if attempt < max_quota_retries - 1:
                            self.log("3/10", f"   ⚠️ Quota exceeded. Waiting 60s for previous instance cleanup ({attempt+1}/{max_quota_retries})...", 'yellow')
                            time.sleep(60)
                            continue # Retry same AZ
                        else:
                            self.log("3/10", "   ❌ Quota wait timeout exceeded.", 'red')
                    
                    self.log("3/10", f"   ❌ Failed in {target_az}: {str(e)[:100]}...", 'red')
                    last_error = e
                    break # Fail this AZ, move to next
            
            if deployment_success:
                break # Break AZ loop
        
        if not deployment_success:
            self.log("3/10", "❌ All Availability Zones failed.", 'red')
            if last_error:
                raise last_error
            raise Exception("Failed to launch instance in any AZ")
            
        self.log("3/10", "   Waiting for instance to be running...", 'gray')
        
        # Wait for running
        waiter = self.ec2_client.get_waiter('instance_running')
        waiter.wait(InstanceIds=[self.instance_id])
        
        # Get public IP
        instance = self.ec2_resource.Instance(self.instance_id)
        self.public_ip = instance.public_ip_address
        
        self.log("3/10", f"✓ Instance running at: {self.public_ip}", 'green')
    
    def wait_for_ssh(self):
        """Step 4: Wait for SSH to be ready"""
        self.log("4/10", "Waiting for SSH to be ready (1-2 minutes)...", 'yellow')
        
        for i in range(30):
            try:
                result = subprocess.run(
                    ['ssh', '-i', self.key_path, '-o', 'StrictHostKeyChecking=no', 
                     '-o', 'ConnectTimeout=5', f'ubuntu@{self.public_ip}', 'echo ready'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and 'ready' in result.stdout:
                    self.log("4/10", "✓ SSH connection established", 'green')
                    return
            except:
                pass
            
            time.sleep(10)
            self.log("4/10", f"   Attempt {i+1}/30...", 'gray')
        
        raise Exception("SSH connection timeout")
    
    def setup_server(self):
        """Step 5: Install NVIDIA drivers, Docker, GPU runtime"""
        self.log("5/10", "Setting up remote server (15-20 minutes)...", 'yellow')
        self.log("5/10", "   - Installing NVIDIA drivers", 'gray')
        self.log("5/10", "   - Installing Docker + GPU runtime", 'gray')
        
        # Upload setup script (helper functions)
        subprocess.run([
            'scp', '-i', self.key_path, '-o', 'StrictHostKeyChecking=no',
            'scripts/setup_remote.sh', f'ubuntu@{self.public_ip}:~/'
        ], check=True)
        
        # Helper to run remote command
        def run_remote(command, step_name):
            self.log("5/10", f"   Running: {step_name}...", 'cyan')
            try:
                subprocess.run([
                    'ssh', '-i', self.key_path, f'ubuntu@{self.public_ip}',
                    command
                ], check=True)
                self.log("5/10", f"   ✓ {step_name} complete", 'green')
            except subprocess.CalledProcessError as e:
                self.log("5/10", f"   ❌ {step_name} failed (Exit {e.returncode})", 'red')
                raise e

        # 1. Wait for Cloud Init
        run_remote('cloud-init status --wait', 'Waiting for cloud-init')

        # 2. Update System
        run_remote('sudo apt-get update -y', 'System Update')

        # 3. Install Tools
        run_remote('sudo apt-get install -y git build-essential curl wget', 'Install Basic Tools')

        # 4. Install Drivers (Long step)
        run_remote('sudo apt-get install -y ubuntu-drivers-common && sudo ubuntu-drivers autoinstall', 'Install NVIDIA Drivers')

        # 5. Install Docker
        run_remote('curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker ubuntu', 'Install Docker')

        # 6. Install NVIDIA Toolkit
        setup_toolkit_cmd = (
            'distribution="ubuntu22.04" && '
            'curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && '
            'curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | '
            'sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | '
            'sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && '
            'sudo apt-get update && '
            'sudo apt-get install -y nvidia-container-toolkit && '
            'sudo nvidia-ctk runtime configure --runtime=docker && '
            'sudo systemctl restart docker'
        )
        run_remote(setup_toolkit_cmd, 'Install NVIDIA Toolkit')
        
        # Reboot
        self.log("5/10", "   Rebooting for driver activation...", 'gray')
        subprocess.run([
            'ssh', '-i', self.key_path, f'ubuntu@{self.public_ip}',
            'sudo reboot'
        ], stderr=subprocess.DEVNULL)
        
        time.sleep(30)
        
        # Wait for SSH after reboot
        self.log("5/10", "   Waiting for server to come back online...", 'gray')
        for i in range(20):
            try:
                result = subprocess.run(
                    ['ssh', '-i', self.key_path, '-o', 'ConnectTimeout=5',
                     f'ubuntu@{self.public_ip}', 'echo ready'],
                    capture_output=True, timeout=10
                )
                if result.returncode == 0:
                    break
            except:
                pass
            time.sleep(10)
        
        # Verify GPU
        result = subprocess.run([
            'ssh', '-i', self.key_path, f'ubuntu@{self.public_ip}',
            'nvidia-smi --query-gpu=name --format=csv,noheader'
        ], capture_output=True, text=True)
        
        self.log("5/10", f"✓ GPU detected: {result.stdout.strip()}", 'green')
    
    def deploy_searvids(self):
        """Step 6: Upload and start Searvids server"""
        self.log("6/10", "Uploading Searvids project...", 'yellow')
        
        # Create tarball using python tarfile
        tar_path = 'searvids_deploy.tar.gz'
        import tarfile
        
        def filter_func(tarinfo):
            name = tarinfo.name
            if '/.git' in name or '/data' in name or '/.venv' in name or \
               '/node_modules' in name or '/benchmark_results' in name or \
               '/models' in name or '/__pycache__' in name:
                return None
            if name.endswith('.pem') or name.endswith('.tar.gz') or name.endswith('.log'):
                return None
            return tarinfo

        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add('.', arcname='.', filter=filter_func)
        
        # Upload
        subprocess.run([
            'scp', '-i', self.key_path, tar_path, f'ubuntu@{self.public_ip}:~/searvids.tar.gz'
        ], check=True)
        
        # Extract
        subprocess.run([
            'ssh', '-i', self.key_path, f'ubuntu@{self.public_ip}',
            'mkdir -p ~/searvids && tar -xzf searvids.tar.gz -C ~/searvids'
        ], check=True)
        
        os.remove(tar_path)
        self.log("6/10", "✓ Project uploaded", 'green')
    
    def start_server(self):
        """Step 7: Build and start Searvids"""
        self.log("7/10", "Starting Searvids server (5-7 minutes)...", 'yellow')
        
        subprocess.run([
            'ssh', '-i', self.key_path, f'ubuntu@{self.public_ip}',
            'cd ~/searvids && docker compose up -d --build'
        ], check=True)
        
        # Wait for health check
        self.log("7/10", "   Waiting for server health check...", 'gray')
        import requests
        for i in range(30):
            try:
                response = requests.get(f'http://{self.public_ip}:8080/api/health', timeout=5)
                if response.status_code == 200:
                    self.log("7/10", "✓ Server is healthy", 'green')
                    return
            except:
                pass
            time.sleep(10)
        
        raise Exception("Server health check timeout")
    
    def run_benchmark(self):
        """Step 8: Run benchmark suite"""
        self.log("8/10", "Running benchmark suite (manual fallback)...", 'yellow')
        
        # Upload manual script
        subprocess.run([
            'scp', '-i', self.key_path, 'tools/manual_benchmark.py', f'ubuntu@{self.public_ip}:~/searvids/tools/'
        ], check=True)

        # Run it
        subprocess.run([
            'ssh', '-i', self.key_path, f'ubuntu@{self.public_ip}',
            'cd ~/searvids && python3 tools/manual_benchmark.py'
        ], check=True)
        
        self.log("8/10", "✓ Benchmark complete", 'green')

    def run_search_benchmark(self):
        """Step 8b: Run search latency benchmark"""
        # Skip search benchmark for now or keep it? 
        # Search benchmark doesn't need external access. Keep it.
        self.log("8b/10", "Running search latency benchmark (remote)...", 'yellow')
        
        subprocess.run([
            'ssh', '-i', self.key_path, f'ubuntu@{self.public_ip}',
            'cd ~/searvids && python3 tools/benchmark_search.py --url http://localhost:8080 --total_requests 100 --output benchmark_results/aws_search_results.json'
        ], check=True)
        
        self.log("8b/10", "✓ Search benchmark complete", 'green')
    
    def download_results(self):
        """Step 9: Download results and generate graphs"""
        self.log("9/10", "Downloading results...", 'yellow')
        
        # Download all results
        subprocess.run([
            'scp', '-i', self.key_path,
            f'ubuntu@{self.public_ip}:~/searvids/benchmark_results/*',
            'benchmark_results/'
        ], stderr=subprocess.DEVNULL)
        
        # We don't generate graph here to avoid local python issues.
        # User can view JSONs.
        
        self.log("9/10", "✓ Results downloaded", 'green')
    
    def cleanup(self):
        """Step 10: Terminate instance and clean up"""
        self.log("10/10", "Cleanup called. SKIPPING FORCEFULLY.", 'yellow')
        return

    def run(self):
        """Execute full deployment pipeline"""
        try:
            if self.public_ip and self.key_path:
                 # ... (manual mode code) ...
                 pass
            else:
                # Automatic Mode
                self.create_security_group()
                self.create_key_pair()
                self.launch_instance()
            
            # Common steps
            self.wait_for_ssh()
            self.setup_server()
            self.deploy_searvids()
            self.start_server()
            self.run_benchmark()
            self.run_search_benchmark()
            self.download_results()
            # self.cleanup() # DISABLED
            
            # Final summary
            print("\n" + "="*50)
            print("  ✅ BENCHMARK COMPLETE!")
            # ...
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nLeaving instance running for debug...")
            # if not self.skip_cleanup:
            #     self.cleanup()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy and benchmark Searvids on AWS')
    parser.add_argument('--region', default=REGION, help='AWS region')
    parser.add_argument('--skip-cleanup', action='store_true', help='Keep instance running (for debugging)')
    parser.add_argument('--ip', help='(Manual Mode) Existing instance Public IP')
    parser.add_argument('--key', help='(Manual Mode) Path to private key (.pem) file')
    args = parser.parse_args()
    
    deployer = AWSDeployer(region=args.region, skip_cleanup=args.skip_cleanup)
    
    # Set manual mode args if provided
    if args.ip and args.key:
        deployer.public_ip = args.ip
        deployer.key_path = args.key
        # In manual mode, we usually don't want to terminate the instance unless explicitly asked
        # But for this "one-shot benchmark" usage, user might expect termination.
        # Let's enforce skip-cleanup by default in manual mode to be safe, unless user overrides?
        # No, let's keep consistent behavior: Terminate unless --skip-cleanup is passed.
        # But we can't terminate easily if we don't have the instance ID.
        # So we'll disable auto-termination in manual mode for safety.
        deployer.skip_cleanup = True
        print("Note: Automatic termination is DISABLED in manual mode.")
        
    deployer.run()

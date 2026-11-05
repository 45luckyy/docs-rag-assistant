# Deploying to AWS (EC2 + Docker)

App Runner closed to new customers in April 2026, so this guide uses a single small EC2 instance
running the Docker image. Accounts created after 15 July 2025 are on the credit-based Free Tier:
EC2 usage is paid from your sign-up credits (up to $200, valid 12 months; the Free plan itself lasts
6 months). Check the [AWS Free Tier page](https://aws.amazon.com/free/) before you start.

Rough cost: a `t3.micro` running 24/7 plus its public IPv4 address is in the single-digit dollars
a month, taken from credits. Check the EC2 pricing page for your region. Stop the instance when
you're not demoing.

## 0. Set a budget alarm first

Console → **Billing and Cost Management → Budgets → Create budget → Use a template → Zero spend
budget** (or a monthly cost budget of $5), and add your email. Or with the CLI:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws budgets create-budget --account-id "$ACCOUNT_ID" \
  --budget '{"BudgetName":"rag-monthly","BudgetLimit":{"Amount":"5","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"you@example.com"}]}]'
```

## 1. Launch the instance

- AMI: Amazon Linux 2023, type `t3.micro` (1 GB RAM is enough for this index).
- Key pair: create one and keep the `.pem` safe.
- Security group: SSH (22) from **My IP** only; HTTP (80) from anywhere.
- Storage: default 8 GB gp3 is fine.

## 2. Install Docker and run the app

```bash
ssh -i rag.pem ec2-user@<public-ip>

sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user && newgrp docker

# 1 GB RAM is tight while pip builds the image: add swap.
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

git clone https://github.com/45luckyy/docs-rag-assistant.git && cd docs-rag-assistant
cp .env.example .env && chmod 600 .env && nano .env    # add GOOGLE_API_KEY, set MIN_SCORE from your sweep

docker build -t docs-rag .
docker run -d --name rag --restart unless-stopped -p 80:8000 \
  --env-file .env -v rag-data:/app/data docs-rag
docker logs -f rag     # first start embeds the corpus (a few minutes on the free tier)
```

Open `http://<public-ip>/`. The index lives in the `rag-data` volume, so restarts don't re-embed.
To rebuild it: `docker run --rm --env-file .env -v rag-data:/app/data docs-rag python -m app.ingest`
(or delete the volume).

## 3. Optional: keep documents in S3

```bash
aws s3 mb s3://<your-bucket>
aws s3 sync data/corpus s3://<your-bucket>/corpus     # upload
aws s3 sync s3://<your-bucket>/corpus data/corpus     # restore on the instance
```

Give the instance an IAM role with read access to that bucket instead of putting AWS keys on it.

## 4. Security notes

- The API key lives only in `.env` on the instance (never in git or the image). For a step up,
  store it in SSM Parameter Store and read it at start-up.
- `/ask` is rate-limited to 10 requests per minute per IP to protect the free-tier quota.
- Plain HTTP is fine for a demo. For HTTPS, put Caddy in front with a domain name.

## 5. Tear down

```bash
docker rm -f rag
# Console: EC2 → Instances → Terminate. Also release any Elastic IP and delete unused volumes/snapshots.
```

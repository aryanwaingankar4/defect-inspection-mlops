// Azure deployment of the defect-inspection API (ADR-039; TASK-036/038/039).
// Deployed by scripts/azure.py (`make azure-up`, `make azure-deploy`) with
// `az deployment group create` into rg-steelsentry. Incremental and idempotent.
// No secret has a default; secrets come only from the deployer's environment.

targetScope = 'resourceGroup'

@description('Azure region (policy allows uaenorth; backup eastasia).')
param location string = resourceGroup().location

@description('Short prefix for resource names.')
param prefix string = 'steelsentry'

@description('Postgres admin login name (not a secret).')
param pgAdminLogin string

@secure()
@description('Postgres admin password (from AZURE_PG_ADMIN_PASSWORD in .env).')
param pgAdminPassword string

@description('Availability zone for Postgres B1ms (uaenorth lists it in zones 1-3 only).')
param pgZone string = '1'

@description('Postgres database the API logs to.')
param pgDatabaseName string = 'defect'

@description('false: infrastructure only (azure-up). true: also the API Container App (azure-deploy).')
param deployApp bool = false

@description('Image reference by digest, e.g. <acr>.azurecr.io/defect-inspection-api@sha256:...')
param image string = ''

@secure()
param apiKey string = ''

@secure()
param auditSalt string = ''

@secure()
@description('postgresql+psycopg://...?sslmode=require (built by scripts/azure.py).')
param databaseUrl string = ''

var suffix = uniqueString(resourceGroup().id)
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var tags = {
  project: 'steelsentry'
  decision: 'ADR-039'
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${prefix}${suffix}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: false
  }
}

resource pullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-api-pull'
  location: location
  tags: tags
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, pullIdentity.id, acrPullRoleId)
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: pullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource pg 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: '${prefix}-pg-${suffix}'
  location: location
  tags: tags
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '16'
    availabilityZone: pgZone
    administratorLogin: pgAdminLogin
    administratorLoginPassword: pgAdminPassword
    storage: {
      storageSizeGB: 32
      autoGrow: 'Disabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: { mode: 'Disabled' }
    network: { publicNetworkAccess: 'Enabled' }
    authConfig: {
      passwordAuth: 'Enabled'
      activeDirectoryAuth: 'Disabled'
    }
  }
}

resource pgDb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: pg
  name: pgDatabaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

resource pgTls 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2024-08-01' = {
  parent: pg
  name: 'require_secure_transport'
  properties: {
    value: 'on'
    source: 'user-override'
  }
  dependsOn: [pgDb]
}

// Only Azure services (0.0.0.0). The deployer IP is added and removed around migrations
// by scripts/azure.py, never here. No VNet: trade-off in ADR-039.
resource pgAzure 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: pg
  name: 'AllowAllAzureServicesAndResourcesWithinAzureIps'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
  dependsOn: [pgTls]
}

resource api 'Microsoft.App/containerApps@2024-03-01' = if (deployApp) {
  name: '${prefix}-api'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${pullIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: pullIdentity.id
        }
      ]
      secrets: [
        { name: 'api-key', value: apiKey }
        { name: 'audit-salt', value: auditSalt }
        { name: 'database-url', value: databaseUrl }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'API_KEY', secretRef: 'api-key' }
            { name: 'AUDIT_SALT', secretRef: 'audit-salt' }
            { name: 'DATABASE_URL', secretRef: 'database-url' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 3
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
  dependsOn: [acrPull, pgAzure]
}

output acrName string = acr.name
output acrLoginServer string = acr.properties.loginServer
output pgServerName string = pg.name
output pgHost string = pg.properties.fullyQualifiedDomainName
output envName string = env.name
output appName string = '${prefix}-api'

// Postgres Flexible Server for the prediction log (ADR-039). Used as a module by main.bicep and
// deployed on its own by `scripts/azure.py pg-probe` to find a region that offers B1ms.
targetScope = 'resourceGroup'

param location string
param serverName string
param adminLogin string

@secure()
param adminPassword string

param databaseName string = 'defect'

@description('Availability zone; empty = no preference.')
param zone string = ''

param tags object = {}

resource pg 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: serverName
  location: location
  tags: tags
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '16'
    availabilityZone: empty(zone) ? null : zone
    administratorLogin: adminLogin
    administratorLoginPassword: adminPassword
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

resource db 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: pg
  name: databaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

resource tls 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2024-08-01' = {
  parent: pg
  name: 'require_secure_transport'
  properties: {
    value: 'on'
    source: 'user-override'
  }
  dependsOn: [db]
}

// Only Azure services (0.0.0.0). The deployer IP is added and removed around migrations
// by scripts/azure.py, never here. No VNet: trade-off in ADR-039.
resource azureServices 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: pg
  name: 'AllowAllAzureServicesAndResourcesWithinAzureIps'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
  dependsOn: [tls]
}

output name string = pg.name
output host string = pg.properties.fullyQualifiedDomainName
output location string = pg.location

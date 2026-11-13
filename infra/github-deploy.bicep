// GitHub Actions deploy identity (ADR-040; TASK-041). OIDC only: no client secret exists.
// A user-assigned managed identity trusts this repo's main branch through a federated
// credential and gets only the rights the deploy workflow needs, scoped to this resource group.
// Deployed by `make azure-github-oidc` (scripts/azure.py github-setup).
targetScope = 'resourceGroup'

param location string
param acrName string
param pullIdentityName string

@description('OIDC subjects to trust, e.g. repo:<owner>/<repo>:ref:refs/heads/main')
param subjects array

var acrPushRoleId = '8311e382-0749-4cb8-b61a-304f252e45ec'
var containerAppsContributorRoleId = '358470bc-b998-42bd-ab17-a7e34c199c0f'
var managedIdentityOperatorRoleId = 'f1a07417-d97a-45cb-824c-7a7467783830'

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource pullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: pullIdentityName
}

resource deployIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'steelsentry-gh-deploy'
  location: location
  tags: {
    project: 'steelsentry'
    decision: 'ADR-040'
  }
}

// Federated credentials must be written one at a time.
@batchSize(1)
resource federated 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = [
  for (subject, i) in subjects: {
    parent: deployIdentity
    name: 'github-${i}'
    properties: {
      issuer: 'https://token.actions.githubusercontent.com'
      subject: subject
      audiences: ['api://AzureADTokenExchange']
    }
  }
]

resource acrPush 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, deployIdentity.id, acrPushRoleId)
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPushRoleId)
    principalId: deployIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource appsContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, deployIdentity.id, containerAppsContributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      containerAppsContributorRoleId
    )
    principalId: deployIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource identityOperator 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(pullIdentity.id, deployIdentity.id, managedIdentityOperatorRoleId)
  scope: pullIdentity
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      managedIdentityOperatorRoleId
    )
    principalId: deployIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

output clientId string = deployIdentity.properties.clientId

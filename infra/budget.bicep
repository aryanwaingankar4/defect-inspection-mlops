// Monthly cost budget on the resource group (ADR-039). Alerts the subscription Owner role,
// so no email address is in the repo. Deployed by `make azure-up`; a failure only warns.
targetScope = 'resourceGroup'

param amount int = 10

@description('First day of the current month, yyyy-MM-01 (set by scripts/azure.py).')
param startDate string

resource budget 'Microsoft.Consumption/budgets@2023-05-01' = {
  name: 'steelsentry-monthly'
  properties: {
    category: 'Cost'
    amount: amount
    timeGrain: 'Monthly'
    timePeriod: { startDate: startDate }
    notifications: {
      actual50: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 50
        contactRoles: ['Owner']
        contactEmails: []
      }
      actual80: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 80
        contactRoles: ['Owner']
        contactEmails: []
      }
      actual100: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        contactRoles: ['Owner']
        contactEmails: []
      }
    }
  }
}

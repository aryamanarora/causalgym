# coref

## Data

| Name | Description | Example |
| :--- | :---------- | :------ |
| `agreement_number_subj-relc` | Subject-verb number agreement across relative clause modifying the subject. | The **guard/guards** that hated the manager **knows/know** |
| `agreement_number_obj-relc` | | The **guard/guards** that the customers hated **knows/know** |
| `agreement_number_pp` | | The **guard/guards** behind the managers **knows/know** |
| `agreement_number_reflexive_subj-relc` | Subject-object number agreement given a reflexive object, across a relative clause modifying the subject. | The **farmer/farmers** that loved the actors embarrassed **himself/themselves** |
| `agreement_number_reflexive_obj-relc` | | The **farmer/farmers** that the actors loved embarrassed **himself/themselves** |
| `agreement_number_reflexive_pp` | | The **farmer/farmers** behind the actors embarrassed **himself/themselves** |
| `fillergap_subj` | Wh-extraction of the subject prohibits existential-there subject clause. | Our neighbour reminded us **who/that** **did/there**. |
| `fillergap_obj-him` | Wh-extraction of object prevents object after verb. | Our neighbor reminded us **who/that** our new friend killed **when/him** |
| `fillergap_obj-it` | Same as above but with inanimate object. | Our neighbor reminded us **what/that** our new friend grabbed **when/it** |
| `fillergap_passive_subj-pp` | Wh-extraction of passive subject. | Our neighbor reminded us **why/who** the farmer was killed by **them/.** |
| `fillergap_ditransitive_recipient` | Wh-extraction of recipient of ditransitive verb. | Our neighbor reminded us **that/who** the farmer showed the box to **them/.** |
| `fillergap_ditransitive_time` | Wh-extraction of time adjunct. | Our neigbor reminded us **that/when** the farmer showed the box to them **today/.** |
| `npi_obj-relc` | | **No/The** consultant that the taxi driver has helped has shown **any/some** |
| `passive` | | The farmer **had/was** killed **him/by**. |

## Deprecated
- `winograd.txt` is from [this repo](https://github.com/salesforce/decaNLP/blob/master/local_data/schema.txt)
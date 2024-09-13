import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { Table } from 'primeng/table';
import { MessageService, ConfirmationService } from 'primeng/api';
import { ContainerService } from 'src/app/demo/service/container.service';
import { MenuItem } from 'primeng/api';




interface Container {
    id?: number;
    code?: string;
    date_time?:string;
    image_input?:string;
    image_output?:string;
    detection_threshold?: number;
  
    
}

@Component({
    templateUrl: './crud.component.html',
    providers: [MessageService, ConfirmationService]
})
export class CrudComponent implements OnInit {

    containers: Container[] = [];
    containerDialog: boolean = false;
    deleteContainerDialog: boolean = false;
    deleteContainersDialog: boolean = false;
    container: Container = {};
    selectedContainers: Container[] = [];
    submitted: boolean = false;
    cols: any[] = [];
    exportColumns: any[] = [];
    displayDialog: boolean = false;
    selectedImage: string = '';
    items: MenuItem[];
    home: MenuItem;
    
    constructor(private containerService: ContainerService, private messageService: MessageService, private confirmationService: ConfirmationService) {
        this.home = {icon: 'pi pi-home', routerLink: '/'};
        this.items = [
            {label: 'Containers', routerLink: '/pages/crud'},
         
          ];
     }
 
    ngOnInit() {
        this.containerService.getContainers().subscribe(data => this.containers = data);
       
        this.cols = [
            { field: 'code', header: 'Code' },
            { field: 'date_time', header: 'Date Time' },
            { field: 'image_path', header: 'Image Path' },
            { field: 'detection_threshold', header: 'Detection Threshold' }
        ];
        
        this.exportColumns = this.cols.map(col => ({ title: col.header, dataKey: col.field }));
      
    }

    isDisabled(threshold: number): boolean {
        return threshold >= 0.95;
      }

    getStatusClass(threshold: number): string {
        return threshold >= 0.95 ? 'success' : 'warning';
      }
    
      getStatus(threshold: number): string {
        return threshold >= 0.95 ? 'Verified' : 'Check It';
      }

    deleteSelectedContainers() {
        this.deleteContainersDialog = true;
    }

    editContainer(container: Container) {
        this.container = { ...container };
        this.containerDialog = true;
    }

    deleteContainer(container: Container) {
        this.deleteContainerDialog = true;
        this.container = { ...container };
    }

    confirmDelete() {
        this.containerService.deleteContainer(this.container.id).subscribe(
            () => {
                this.containers = this.containers.filter(val => val.id !== this.container.id);
                this.messageService.add({
                    severity: 'success', 
                    summary: 'Successful', 
                    detail: 'Container Deleted', 
                    life: 3000 
                });
                this.deleteContainerDialog = false;
                this.container = {};
            },
            (error) => {
                this.messageService.add({
                    severity: 'error', 
                    summary: 'Error', 
                    detail: 'Failed to delete container', 
                    life: 3000 
                });
            }
        );
    }
    
    confirmDeleteSelected() {
        this.containers = this.containers.filter(val => !this.selectedContainers.includes(val));
        this.messageService.add({ severity: 'success', summary: 'Successful', detail: 'Containers Deleted', life: 3000 });
        this.deleteContainersDialog = false;
        this.selectedContainers = [];
    }

    hideDialog() {
        this.containerDialog = false;
        this.submitted = false;
    }

    ConfirmContainer(container: any) {
        if (!container.id) {
            this.messageService.add({ severity: 'error', summary: 'Error', detail: 'Container ID is missing', life: 3000 });
            return;
        }
    
        const updatedContainer = {
            id: container.id,
            code: container.code,
            detection_threshold: 0.95
        };
        
        this.containerService.updateContainer(updatedContainer).subscribe(
            response => {
                this.messageService.add({ severity: 'success', summary: 'Successful', detail: 'Container Confirmed', life: 3000 });
                this.containers = this.containers.map(c => c.id === container.id ? { ...c, ...updatedContainer } : c);
                this.containerDialog = false;
            },
            error => {
                console.error('Error response:', error);
                this.messageService.add({ severity: 'error', summary: 'Error', detail: 'Error confirming container', life: 3000 });
            }
        );
        
       
    }
    

    onGlobalFilter(table: Table, event: Event) {
        table.filterGlobal((event.target as HTMLInputElement).value, 'contains');
    }
    showImage(imageUrl: string) {
        this.selectedImage = imageUrl;
        this.displayDialog = true;
    }
}

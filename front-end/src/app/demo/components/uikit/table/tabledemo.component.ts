import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { Table } from 'primeng/table';
import { MessageService, ConfirmationService } from 'primeng/api';
import { ContainerService } from 'src/app/demo/service/container.service';

interface ExpandedRows {
    [key: string]: boolean;
}

@Component({
    templateUrl: './tabledemo.component.html',
    providers: [MessageService, ConfirmationService]
})
export class TableDemoComponent implements OnInit {

    containers: any[] = [];
   
  
    @ViewChild('filter') filter!: ElementRef;

    constructor(private containerService: ContainerService) { }
  
    ngOnInit(): void {
        this.loadContainers();
    }
  
    loadContainers(): void {
        this.containerService.getContainers().subscribe(data => {
            this.containers = data;
        });
    }
  

    onGlobalFilter(table: Table, event: Event) {
        table.filterGlobal((event.target as HTMLInputElement).value, 'contains');
    }

    clear(table: Table) {
        table.clear();
        this.filter.nativeElement.value = '';
    }
}
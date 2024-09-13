import { Component, OnInit } from '@angular/core';
import { MessageService, ConfirmationService } from 'primeng/api';
import { LayoutService } from 'src/app/layout/service/app.layout.service';
import { ContainerService } from '../../service/container.service';
export interface container {
    id: number;
    code: string;
    date_time:string;
    detection_threshold: number;
    image_input: File | null;
    image_output:string| null;
    
  
  }

@Component({
    templateUrl: './dashboard.component.html',
    providers: [MessageService, ConfirmationService]
})
export class DashboardComponent implements OnInit {
    container:container[];
    containersCount: number = 0;
    containersUnderThresholdCount: number = 0;
    containersAboveThresholdCount: number = 0;
    threshold: number = 0.95;
    selectedFile: File | null = null;
    containerData:container | null= null;
    containerVideoData:container| null=null;

    constructor( private containerService:ContainerService,private layoutService:LayoutService, private messageService: MessageService) {
     
    }

    ngOnInit() {
        this.loadContainerCount(); 
        this.fetchLatestData();
    }

    loadContainerCount(): void {
        this.containerService.getContainers().subscribe(data => {
     
            const under=data.filter(container => container.detection_threshold < this.threshold).length;
            const above=data.filter(container => container.detection_threshold >= this.threshold).length;
            this.containersUnderThresholdCount=under;
            this.containersAboveThresholdCount=above;
            this.containersCount = data.length; // Suppose que tu as un service qui renvoie une liste des containers  
        });
    }

    fetchLatestData(): void {
        this.containerService.getLatestData().subscribe(data => {
          this.containerVideoData = data;
        });}


    onFileSelected(event: Event): void {
        const input = event.target as HTMLInputElement;
        if (input.files && input.files.length > 0) {
            this.selectedFile = input.files[0];
            
        }
    }

    

  onUpload(): void {
    if (this.selectedFile) {
        const formData = new FormData();
        formData.append('image', this.selectedFile, this.selectedFile.name);

        this.containerService.uploadContainerImage(formData).subscribe(
            response => {
                
                console.log('Upload successful:', response);
                this.containerData=response;
                this.messageService.add({
                    severity: 'success',
                    summary: 'Successful',
                    detail: 'Image uploaded successfully and processed',
                    life: 3000
                });
                this.loadContainerCount(); // Recharger les données après l'upload
            },
            error => {
                console.error('Upload failed:', error);
                this.messageService.add({
                    severity: 'error',
                    summary: 'Error',
                    detail: 'Image upload failed',
                    life: 3000
                });
            }
        );
    } else {
        this.messageService.add({
            severity: 'warn',
            summary: 'Warning',
            detail: 'No file selected',
            life: 3000
        });
    }
}

       

          
        

}

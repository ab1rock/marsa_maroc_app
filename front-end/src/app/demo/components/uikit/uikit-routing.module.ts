import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';

@NgModule({
    imports: [RouterModule.forChild([
        
        { path: 'charts', data: { breadcrumb: 'Charts' }, loadChildren: () => import('./charts/chartsdemo.module').then(m => m.ChartsDemoModule) },

    
        { path: 'media', data: { breadcrumb: 'Media' }, loadChildren: () => import('./media/mediademo.module').then(m => m.MediaDemoModule) },

       
        { path: 'table', data: { breadcrumb: 'Table' }, loadChildren: () => import('./table/tabledemo.module').then(m => m.TableDemoModule) },
    
      
        
        { path: '**', redirectTo: '/notfound' }
    ])],
    exports: [RouterModule]
})
export class UIkitRoutingModule { }
